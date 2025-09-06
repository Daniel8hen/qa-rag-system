# Project Overview

This project is a Q&A RAG application that allows upload documents / send URLs, via CLI, and then they can ask questions and get answers based on it.
It is a robust and scalable pipeline designed to perform efficient question answering over large collections of documents from multiple sources.  
Using **OpenAI embeddings**, **ChromaDB**, **LangChain**, and **advanced web scraping**, this system can process PDFs, web pages, and mixed document sources to create comprehensive knowledge bases.

## Folder Structure


- src/models/ → LLM and embedding model initialization (e.g., model_generator.py)
- src/utils/ → Document loaders, text processing utilities (e.g., document_loader.py, text_processer.py)
- src/data_interactor.py → Vector store operations with ChromaDB
- src/main.py → CLI entry point for processing and querying

- tests/ → Unit tests

When adding new functionality:
- Place model-related logic in src/models/.
- Place utilities (parsers, loaders, processors) in src/utils/.
- Extend CLI options in src/main.py.
## Coding Standards

- Language: Python 3.x
- Follow PEP8 conventions.
- Write modular, reusable functions with clear docstrings.
- Use async/await for concurrent operations where applicable.
- Handle errors gracefully (e.g., failed downloads, malformed documents).
- Always add unit tests for new modules or functions.

## CLI guidelines

- This project is CLI-driven
- Provide clear console messages when:
    - Processing files/URLs
    - Skipping duplicates
    - Querying results
- Output should be concise, informative, and user-friendly.

## Testing Guidance
- Use pytest for tests (pytest tests/).
- Cover:
    - Document processing (PDFs, web, mixed)
    - ChromaDB interactions
    - Query flow (retrieval + generation)
    - Provide sample JSON or mock data where possible.