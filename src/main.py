import argparse
import logging
from data_interactor import generate_chroma_db_from_docs, generate_retriever_chain, ask
from models.model_generator import wrapper_emb_llm

def main():
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Argument parser
    parser = argparse.ArgumentParser(description="CLI for processing PDFs or querying the retriever.")
    parser.add_argument("--action", choices=["process", "query"], required=True, help="Specify the action to perform (process or query).")
    parser.add_argument("--pdf_path", type=str, help="Provide the path to the PDF (required for process).")
    parser.add_argument("--question", type=str, help="Provide the question to ask (required for query).")

    # parsing the arguments from the script
    args = parser.parse_args()

    # Generate embedding and LLM models
    embeddings_model, llm = wrapper_emb_llm()
    logger.info("Embedding and LLM models loaded successfully.")

    if args.action == "process":
        if not args.pdf_path:
            parser.error("--pdf_path required for 'process' action.")

        # Generate Chroma DB from PDF
        # pdf_path = "data/pdf/datascience_paper.pdf"
        logger.info(f"Generating Chroma DB from {args.pdf_path}.")
        generate_chroma_db_from_docs(embeddings_model, args.pdf_path)
        logger.info("Chroma DB generated successfully.")

    elif args.action == "query":
        if not args.question:
            parser.error("--question required for 'query' action.")

        # Set up retriever and QA chain
        logger.info("Setting up retriever and QA chain.")
        chain = generate_retriever_chain(embeddings_model, llm)

        # Ask a question
        # question = "What is feature engineering? Please write a full paragraph"
        answer = ask(args.question, chain=chain)
        logger.info(f"{answer}")

if __name__ == "__main__":
    main()