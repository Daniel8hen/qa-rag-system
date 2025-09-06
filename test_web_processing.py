#!/usr/bin/env python3
"""
Test script for web document processing functionality
"""

import sys
import os
import asyncio
import logging
import pytest
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils.document_loader import WebDocumentLoader, BatchDocumentLoader, create_document_loader
from utils.text_processer import generate_chunks_from_urls
from models.model_generator import wrapper_emb_llm
from data_interactor import generate_chroma_db_from_urls, generate_retriever_chain, ask

@pytest.mark.asyncio
async def test_document_loader():
    """Test the basic document loading functionality"""
    print("🧪 Testing WebDocumentLoader...")
    
    # Test with a simple webpage
    test_urls = [
        "https://httpbin.org/html",
        "https://jsonplaceholder.typicode.com/posts/1"  # This will be JSON, should fail gracefully
    ]
    
    for url in test_urls:
        print(f"Testing URL: {url}")
        loader = WebDocumentLoader(url)
        documents = await loader.load()
        
        if documents:
            doc = documents[0]
            print(f"✅ Successfully loaded: {doc.metadata.get('title', 'Untitled')}")
            print(f"   Content length: {len(doc.page_content)} chars")
            print(f"   Content hash: {doc.metadata.get('content_hash', 'N/A')}")
        else:
            print(f"❌ Failed to load content from {url}")
        print()

def test_batch_processing():
    """Test batch processing with multiple URLs"""
    print("🔄 Testing batch URL processing...")
    
    test_urls = [
        "https://httpbin.org/html",
        "https://httpbin.org/robots.txt"
    ]
    
    try:
        chunks = generate_chunks_from_urls(test_urls, chunk_size=500, max_concurrent=2)
        print(f"✅ Successfully processed {len(chunks)} chunks from {len(test_urls)} URLs")
        
        for i, chunk in enumerate(chunks[:2]):  # Show first 2 chunks
            print(f"   Chunk {i+1}:")
            print(f"   - Source: {chunk.metadata.get('source_url', 'Unknown')}")
            print(f"   - Length: {len(chunk.page_content)} chars")
            print(f"   - Preview: {chunk.page_content[:100]}...")
            print()
            
    except Exception as e:
        print(f"❌ Batch processing failed: {e}")
        print()

def test_end_to_end():
    """Test the complete end-to-end processing"""
    print("🚀 Testing end-to-end web processing pipeline...")
    
    try:
        # Load models
        print("Loading models...")
        embeddings_model, llm = wrapper_emb_llm()
        
        # Test URLs
        test_urls = ["https://httpbin.org/html"]
        
        print(f"Processing {len(test_urls)} URLs...")
        generate_chroma_db_from_urls(embeddings_model, test_urls, max_concurrent=1)
        
        print("Setting up QA chain...")
        chain = generate_retriever_chain(embeddings_model, llm)
        
        print("Testing query...")
        test_questions = [
            "What is the title of the webpage?",
            "Describe the content that was processed."
        ]
        
        for question in test_questions:
            print(f"\nQuestion: {question}")
            answer = ask(question, chain)
            print(f"Answer: {answer}")
        
        print("✅ End-to-end test completed successfully!")
        
    except Exception as e:
        print(f"❌ End-to-end test failed: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """Run all tests"""
    print("🌐 Starting Web Document Processing Tests\n")
    print("=" * 60)
    
    # Test 1: Basic document loading
    await test_document_loader()
    print("=" * 60)
    
    # Test 2: Batch processing
    test_batch_processing()
    print("=" * 60)
    
    # Test 3: End-to-end processing
    test_end_to_end()
    
    print("\n🎉 All tests completed!")

if __name__ == "__main__":
    asyncio.run(main())