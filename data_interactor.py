from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from text_processer import generate_chunks_from_pdf

persist_dir = "text_index"


def generate_chroma_db_from_docs(
    embeddings_model, pdf_path: str = "poc_emb/datascience_paper.pdf"
):
    """
    Create and persist a Chroma database from a PDF document.
    :param embeddings_model: The embedding model to use.
    :param pdf_path: Path to the PDF file.
    """
    chunks = generate_chunks_from_pdf(pdf_path)
    db = Chroma.from_documents(
        chunks, embedding=embeddings_model, persist_directory=persist_dir
    )
    # db.persist()


def generate_retriever_chain(embeddings_model, llm, top_k: int = 3):
    """
    Generate a retriever and a QA chain for answering questions.
    :param embeddings_model: The embedding model to use.
    :param llm: The language model for answering queries.
    :param top_k: Number of documents to retrieve.
    :return: Tuple of retriever and QA chain.
    """
    vectordb = Chroma(
        persist_directory=persist_dir, embedding_function=embeddings_model
    )

    # Create a retriever
    retriever = vectordb.as_retriever(search_kwargs={"k": top_k})

    # Define a custom prompt template for the QA chain
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "Use the following context to answer the question.\n"
            "Summaries: {summaries}\n"
            "Question: {question}\n\nAnswer:"
        ),
    )

    # Create the RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
    )

    return retriever, qa_chain


def ask(question: str, retriever, chain) -> str:
    """
    Answer a user question based on the retriever and chain.
    :param question: The user's question.
    :param retriever: The document retriever.
    :param chain: The QA chain.
    :return: The answer as a string.
    """
    # Retrieve the context and get the answer
    response = chain.invoke({"query": question})
    answer = response["result"]

    return f"Answer: {answer}\n\n"
