from langchain_chroma import Chroma
from langchain.chains import RetrievalQA # TODO remove
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from utils.text_processer import generate_chunks_from_pdf, generate_chunks_from_urls, generate_chunks_from_sources
from typing import List, Union
import asyncio
import logging

logger = logging.getLogger(__name__)

PERSIST_DIR = "text_index"

def generate_chroma_db_from_docs(
    embeddings_model, pdf_path: str = "poc_emb/data/pdf/datascience_paper.pdf"
):
    """
    Create and persist a Chroma database from a PDF document.
    :param embeddings_model: The embedding model to use.
    :param pdf_path: Path to the PDF file.
    """
    chunks = generate_chunks_from_pdf(pdf_path)
    db = Chroma.from_documents(
        chunks, embedding=embeddings_model, persist_directory=PERSIST_DIR
    )

def generate_chroma_db_from_urls(
    embeddings_model, 
    urls: List[str], 
    max_concurrent: int = 5,
    chunk_size: int = 4000,
    chunk_overlap: int = 20
):
    """
    Create and persist a Chroma database from web URLs.
    :param embeddings_model: The embedding model to use.
    :param urls: List of URLs to process.
    :param max_concurrent: Maximum concurrent downloads.
    :param chunk_size: Maximum chunk size for splitting.
    :param chunk_overlap: Overlap size between chunks.
    """
    logger.info(f"Processing {len(urls)} URLs for ChromaDB creation")
    chunks = generate_chunks_from_urls(urls, chunk_size, chunk_overlap, max_concurrent)
    
    if not chunks:
        raise ValueError("No content could be extracted from provided URLs")
    
    logger.info(f"Creating ChromaDB with {len(chunks)} chunks")
    db = Chroma.from_documents(
        chunks, embedding=embeddings_model, persist_directory=PERSIST_DIR
    )
    logger.info("ChromaDB created successfully from URLs")
    
def generate_chroma_db_from_sources(
    embeddings_model,
    sources: List[str],
    max_concurrent: int = 5,
    chunk_size: int = 4000,
    chunk_overlap: int = 20
):
    """
    Create and persist a Chroma database from mixed sources (PDFs, URLs, etc.).
    :param embeddings_model: The embedding model to use.
    :param sources: List of source paths/URLs.
    :param max_concurrent: Maximum concurrent processing.
    :param chunk_size: Maximum chunk size for splitting.
    :param chunk_overlap: Overlap size between chunks.
    """
    logger.info(f"Processing {len(sources)} mixed sources for ChromaDB creation")
    
    # Handle async properly
    try:
        # Try to get the current event loop
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No event loop running, safe to use asyncio.run()
        chunks = asyncio.run(generate_chunks_from_sources(
            sources, chunk_size, chunk_overlap, max_concurrent
        ))
    else:
        # We're in an event loop, need to create a task
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(
                lambda: asyncio.run(generate_chunks_from_sources(
                    sources, chunk_size, chunk_overlap, max_concurrent
                ))
            )
            chunks = future.result()
    
    if not chunks:
        raise ValueError("No content could be extracted from provided sources")
    
    logger.info(f"Creating ChromaDB with {len(chunks)} chunks")
    db = Chroma.from_documents(
        chunks, embedding=embeddings_model, persist_directory=PERSIST_DIR
    )
    logger.info("ChromaDB created successfully from mixed sources")


def generate_retriever_chain(embeddings_model, llm, top_k: int = 3):
    """
    Generate a retriever and a QA chain for answering questions.
    :param embeddings_model: The embedding model to use.
    :param llm: The language model for answering queries.
    :param top_k: Number of documents to retrieve.
    :return: Tuple of retriever and QA chain.
    """
    vectordb = Chroma(
        persist_directory=PERSIST_DIR, embedding_function=embeddings_model
    )

    # Create a retriever
    retriever = vectordb.as_retriever(search_kwargs={"k": top_k})

    # Define a custom prompt template for the QA chain
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Use the following context to answer the question.\n"
            "Context: {context}\n"
            "\n\nAnswer:"),
            ("human", "{input}")
        ]
    )

    # Create the RetrievalQA chain
    # Add comment for testing github actions
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    qa_chain = create_retrieval_chain(retriever, question_answer_chain)

    return qa_chain


def ask(question: str, chain) -> str:
    """
    Answer a user question based on the retriever and chain.
    :param question: The user's question.
    :param chain: The QA chain.
    :return: The answer as a string.
    """
    # Retrieve the context and get the answer
    response = chain.invoke({"input": question})
    answer = response["answer"]

    return f"Answer: {answer}\n\n"
