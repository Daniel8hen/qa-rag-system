from typing import List
import asyncio
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from .document_loader import (
    PDFDocumentLoader,
    BatchDocumentLoader,
    create_document_loader,
)

logger = logging.getLogger(__name__)


def generate_chunks_from_pdf(pdf_path: str, chunk_size: int = 4000, chunk_overlap: int = 20) -> List:
    """
    Generates text chunks from a PDF file.
    :param pdf_path: Path to the PDF file.
    :param chunk_size: Maximum chunk size for splitting.
    :param chunk_overlap: Overlap size between chunks.
    :return: List of chunks as "documents" in LangChain jargon.
    """
    # Use the new document loader for consistency
    loader = PDFDocumentLoader(pdf_path)
    documents = asyncio.run(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    return text_splitter.split_documents(documents)


async def generate_chunks_from_sources(
    sources: List[str],
    chunk_size: int = 4000,
    chunk_overlap: int = 20,
    max_concurrent: int = 5,
) -> List[Document]:
    """
    Generates text chunks from multiple sources (PDFs, URLs, etc.).
    :param sources: List of source paths/URLs.
    :param chunk_size: Maximum chunk size for splitting.
    :param chunk_overlap: Overlap size between chunks.
    :param max_concurrent: Maximum concurrent downloads.
    :return: List of chunked documents.
    """
    logger.info(f"Processing {len(sources)} sources with max {max_concurrent} concurrent")

    # Create document loaders
    loaders = []
    for source in sources:
        try:
            loader = create_document_loader(source)
            loaders.append(loader)
        except ValueError as e:
            logger.error(f"Skipping unsupported source {source}: {e}")
            continue

    if not loaders:
        logger.warning("No valid document loaders created")
        return []

    # Load documents concurrently
    batch_loader = BatchDocumentLoader(max_concurrent=max_concurrent)
    documents = await batch_loader.load_documents(loaders)

    if not documents:
        logger.warning("No documents loaded successfully")
        return []

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )

    chunks = text_splitter.split_documents(documents)
    logger.info(f"Generated {len(chunks)} chunks from {len(documents)} documents")

    return chunks


def generate_chunks_from_urls(
    urls: List[str],
    chunk_size: int = 4000,
    chunk_overlap: int = 20,
    max_concurrent: int = 5,
) -> List[Document]:
    """
    Synchronous wrapper for generating chunks from URLs.
    :param urls: List of URLs to process.
    :param chunk_size: Maximum chunk size for splitting.
    :param chunk_overlap: Overlap size between chunks.
    :param max_concurrent: Maximum concurrent downloads.
    :return: List of chunked documents.
    """
    try:
        # Check for an existing event loop; don't assign to an unused variable
        asyncio.get_running_loop()
    except RuntimeError:
        # No event loop running, safe to use asyncio.run()
        return asyncio.run(
            generate_chunks_from_sources(urls, chunk_size, chunk_overlap, max_concurrent)
        )
    else:
        # We're in an event loop, run the coroutine in a thread
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(
                lambda: asyncio.run(
                    generate_chunks_from_sources(urls, chunk_size, chunk_overlap, max_concurrent)
                )
            )
            return future.result()
