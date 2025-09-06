from langchain_chroma import Chroma
from langchain.chains import RetrievalQA # TODO remove
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from utils.text_processer import generate_chunks_from_pdf
from typing import List, Union, Iterable
import asyncio
import logging
from utils.document_loader import create_document_loader, BatchDocumentLoader
from langchain.schema import Document
from pathlib import Path

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


def _run_async(coro):
    """Run coroutine, using asyncio.run unless an event loop is already running."""
    try:
        return asyncio.run(coro)
    except RuntimeError:
        # If there's already a running loop (e.g., in notebooks), use alternative
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)


def _persist_documents_to_chroma(documents: List[Document], embeddings_model):
    """Persist a list of LangChain Documents into ChromaDB using the given embedding model."""
    if not documents:
        logger.warning("No documents to persist to ChromaDB.")
        return None

    db = Chroma.from_documents(documents, embedding=embeddings_model, persist_directory=PERSIST_DIR)
    logger.info(f"Persisted {len(documents)} documents to ChromaDB at {PERSIST_DIR}")
    return db


def generate_chroma_db_from_urls(
    embeddings_model,
    urls: Iterable[str],
    max_concurrent: int = 5,
    chunk_size: int = 4000,
    chunk_overlap: int = 20,
):
    """
    Generate and persist a Chroma DB from a list of URLs.
    :param embeddings_model: embedding model instance
    :param urls: iterable of URLs to load
    :param max_concurrent: concurrency for downloads
    :param chunk_size: chunk size (passed to loaders if applicable)
    :param chunk_overlap: chunk overlap
    :return: Chroma DB instance or None
    """
    # Build document loaders
    loaders = []
    for u in urls:
        try:
            loaders.append(create_document_loader(u))
        except ValueError as e:
            logger.warning(f"Skipping unsupported source {u}: {e}")

    if not loaders:
        logger.error("No valid loaders could be created for provided URLs.")
        return None

    batch_loader = BatchDocumentLoader(max_concurrent=max_concurrent)

    try:
        documents = _run_async(batch_loader.load_documents(loaders))
    except Exception as e:
        logger.error(f"Failed to load documents from URLs: {e}")
        return None

    return _persist_documents_to_chroma(documents, embeddings_model)


def generate_chroma_db_from_sources(
    embeddings_model,
    sources: Iterable[str],
    max_concurrent: int = 5,
    chunk_size: int = 4000,
    chunk_overlap: int = 20,
):
    """
    Generate and persist a Chroma DB from mixed sources (file paths, URLs).
    :param embeddings_model: embedding model instance
    :param sources: iterable of sources (URLs or file paths)
    :param max_concurrent: concurrency for downloads
    :return: Chroma DB instance or None
    """
    loaders = []
    for s in sources:
        try:
            loaders.append(create_document_loader(s))
        except ValueError:
            # if not a direct loader, try local file fallback for PDFs
            if Path(s).exists() and s.lower().endswith('.pdf'):
                loaders.append(create_document_loader(s))
            else:
                logger.warning(f"Unsupported or missing source skipped: {s}")

    if not loaders:
        logger.error("No valid loaders found in sources.")
        return None

    batch_loader = BatchDocumentLoader(max_concurrent=max_concurrent)

    try:
        documents = _run_async(batch_loader.load_documents(loaders))
    except Exception as e:
        logger.error(f"Failed to load documents from sources: {e}")
        return None

    return _persist_documents_to_chroma(documents, embeddings_model)


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
