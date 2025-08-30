#!/usr/bin/env python3
"""
Debug tool to test URL processing and diagnose extraction issues
"""

import sys
import os
import asyncio
import logging

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils.document_loader import WebDocumentLoader

# Set up logging to see debug info
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

async def debug_url(url):
    """Debug a single URL extraction"""
    print(f"🔍 Debugging URL: {url}")
    print("=" * 60)
    
    loader = WebDocumentLoader(url)
    
    try:
        documents = await loader.load()
        
        if documents:
            doc = documents[0]
            print(f"✅ SUCCESS!")
            print(f"   Title: {doc.metadata.get('title', 'No title')}")
            print(f"   Content length: {len(doc.page_content)} characters")
            print(f"   Content hash: {doc.metadata.get('content_hash', 'N/A')}")
            print(f"   Source URL: {doc.metadata.get('source_url', 'N/A')}")
            print(f"\n📝 Content preview (first 300 chars):")
            print("-" * 40)
            print(doc.page_content[:300] + "...")
            print("-" * 40)
        else:
            print("❌ FAILED: No content extracted")
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

async def main():
    if len(sys.argv) != 2:
        print("Usage: python debug_url.py <URL>")
        print("Example: python debug_url.py https://example.com")
        sys.exit(1)
    
    url = sys.argv[1]
    await debug_url(url)

if __name__ == "__main__":
    asyncio.run(main())