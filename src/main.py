import logging
from data_interactor import generate_chroma_db_from_docs, generate_retriever_chain, ask
from src.models.model_generator import wrapper_emb_llm

def main():
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Generate embedding and LLM models
    embeddings_model, llm = wrapper_emb_llm()
    logger.info("Embedding and LLM models loaded successfully.")

    # Generate Chroma DB from PDF
    pdf_path = "../data/pdf/datascience_paper.pdf"
    logger.info(f"Generating Chroma DB from {pdf_path}.")
    generate_chroma_db_from_docs(embeddings_model, pdf_path)

    # Set up retriever and QA chain
    logger.info("Setting up retriever and QA chain.")
    retriever, chain = generate_retriever_chain(embeddings_model, llm)

    # Ask a question
    question = "What is feature engineering? Please write a full paragraph"
    answer = ask(question, retriever=retriever, chain=chain)
    print(answer)

if __name__ == "__main__":
    main()