from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

def generate_chunks_from_pdf(pdf_path: str, chunk_size: int = 4000, chunk_overlap: int = 20) -> List:
    """
    Generates text chunks from a PDF file.
    :param pdf_path: Path to the PDF file.
    :param chunk_size: Maximum chunk size for splitting.
    :param chunk_overlap: Overlap size between chunks.
    :return: List of chunks as "documents" in LangChain jargon.
    """
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    return text_splitter.split_documents(pages)
