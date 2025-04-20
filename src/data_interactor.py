from langchain_chroma import Chroma
from langchain.chains import RetrievalQA # TODO remove
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from utils.text_processer import generate_chunks_from_pdf

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
