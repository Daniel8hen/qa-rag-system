from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path
import asyncio
import aiohttp
import requests
from urllib.parse import urlparse, urljoin
import hashlib
import logging
from datetime import datetime

from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from bs4 import BeautifulSoup
import trafilatura

logger = logging.getLogger(__name__)


class DocumentLoader(ABC):
    """Abstract base class for document loaders."""
    
    @abstractmethod
    async def load(self) -> List[Document]:
        """Load documents from the source."""
        pass
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the document source."""
        pass


class PDFDocumentLoader(DocumentLoader):
    """Loader for PDF documents."""
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
    
    async def load(self) -> List[Document]:
        """Load PDF document."""
        loader = PyPDFLoader(self.pdf_path)
        pages = loader.load_and_split()
        
        # Add metadata
        for page in pages:
            page.metadata.update({
                "source_type": "pdf",
                "source_path": self.pdf_path,
                "processed_at": datetime.now().isoformat()
            })
        
        return pages
    
    def get_metadata(self) -> Dict[str, Any]:
        return {
            "source_type": "pdf",
            "source_path": self.pdf_path,
            "file_exists": Path(self.pdf_path).exists()
        }


class WebDocumentLoader(DocumentLoader):
    """Loader for web documents with content extraction."""
    
    def __init__(self, url: str, extract_method: str = "trafilatura"):
        self.url = url
        self.extract_method = extract_method
        self._content_hash = None
    
    async def load(self) -> List[Document]:
        """Load web document with async HTTP request."""
        try:
            # Create SSL context that doesn't verify certificates for testing
            import ssl
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=60),  # Increased timeout to 60s
                connector=connector,
                headers=headers
            ) as session:
                async with session.get(self.url) as response:
                    if response.status != 200:
                        raise Exception(f"HTTP {response.status} for {self.url}")
                    
                    html_content = await response.text()
                    
            # Extract clean text content
            text_content = None
            extraction_method_used = None
            
            if self.extract_method == "trafilatura":
                text_content = trafilatura.extract(html_content)
                extraction_method_used = "trafilatura"
                
                # If trafilatura fails, try BeautifulSoup as fallback
                if not text_content or len(text_content.strip()) < 100:
                    logger.info(f"Trafilatura extracted minimal content from {self.url}, trying BeautifulSoup fallback")
                    soup = BeautifulSoup(html_content, 'html.parser')
                    
                    # Remove script and style elements
                    for script in soup(["script", "style", "nav", "header", "footer"]):
                        script.decompose()
                    
                    text_content = soup.get_text(separator=' ', strip=True)
                    extraction_method_used = "beautifulsoup_fallback"
            else:
                # Primary BeautifulSoup extraction
                soup = BeautifulSoup(html_content, 'html.parser')
                for script in soup(["script", "style", "nav", "header", "footer"]):
                    script.decompose()
                text_content = soup.get_text(separator=' ', strip=True)
                extraction_method_used = "beautifulsoup"
            
            # Debug logging
            content_length = len(text_content) if text_content else 0
            logger.info(f"Extraction from {self.url}: method={extraction_method_used}, content_length={content_length}")
            
            if not text_content or len(text_content.strip()) < 100:
                logger.warning(f"Very little content extracted from {self.url} (length: {content_length})")
                logger.debug(f"HTML preview: {html_content[:500]}...")
                return []
            
            # Generate content hash for deduplication
            self._content_hash = hashlib.md5(text_content.encode()).hexdigest()
            
            # Create document with metadata
            document = Document(
                page_content=text_content,
                metadata={
                    "source_type": "web",
                    "source_url": self.url,
                    "content_hash": self._content_hash,
                    "processed_at": datetime.now().isoformat(),
                    "title": self._extract_title(html_content),
                    "content_length": len(text_content)
                }
            )
            
            return [document]
            
        except Exception as e:
            error_type = type(e).__name__
            logger.error(f"Failed to load {self.url}: {error_type} - {str(e)}")
            
            # More specific error handling
            if "SSL" in str(e) or "certificate" in str(e).lower():
                logger.error("SSL certificate issue detected")
            elif "timeout" in str(e).lower():
                logger.error("Request timeout")
            elif "403" in str(e) or "Forbidden" in str(e):
                logger.error("Access forbidden - possible anti-bot protection")
            elif "404" in str(e) or "Not Found" in str(e):
                logger.error("URL not found")
            elif "connection" in str(e).lower():
                logger.error("Connection issue")
            
            return []
    
    def _extract_title(self, html_content: str) -> str:
        """Extract page title from HTML."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            title_tag = soup.find('title')
            return title_tag.get_text(strip=True) if title_tag else "Untitled"
        except:
            return "Untitled"
    
    def get_metadata(self) -> Dict[str, Any]:
        return {
            "source_type": "web",
            "source_url": self.url,
            "content_hash": self._content_hash,
            "extract_method": self.extract_method
        }


class BatchDocumentLoader:
    """Batch loader for processing multiple documents concurrently."""
    
    def __init__(self, max_concurrent: int = 5):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def load_documents(self, loaders: List[DocumentLoader]) -> List[Document]:
        """Load multiple documents concurrently."""
        tasks = [self._load_with_semaphore(loader) for loader in loaders]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten results and filter out exceptions
        documents = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Document loading failed: {str(result)}")
            elif isinstance(result, list):
                documents.extend(result)
        
        # Remove duplicates based on content hash
        return self._deduplicate_documents(documents)
    
    async def _load_with_semaphore(self, loader: DocumentLoader) -> List[Document]:
        """Load document with concurrency control."""
        async with self.semaphore:
            return await loader.load()
    
    def _deduplicate_documents(self, documents: List[Document]) -> List[Document]:
        """Remove duplicate documents based on content hash."""
        seen_hashes = set()
        unique_documents = []
        
        for doc in documents:
            content_hash = doc.metadata.get("content_hash")
            if content_hash and content_hash in seen_hashes:
                logger.info(f"Skipping duplicate document: {doc.metadata.get('source_url', 'unknown')}")
                continue
            
            if content_hash:
                seen_hashes.add(content_hash)
            unique_documents.append(doc)
        
        logger.info(f"Processed {len(documents)} documents, {len(unique_documents)} unique after deduplication")
        return unique_documents


def create_document_loader(source: str) -> DocumentLoader:
    """Factory function to create appropriate document loader."""
    if source.startswith(('http://', 'https://')):
        return WebDocumentLoader(source)
    elif source.endswith('.pdf'):
        return PDFDocumentLoader(source)
    else:
        raise ValueError(f"Unsupported document source: {source}")