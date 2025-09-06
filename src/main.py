import logging
import json
import sys
from data_interactor import (
    generate_chroma_db_from_docs,
    generate_chroma_db_from_urls,
    generate_chroma_db_from_sources,
    generate_retriever_chain,
    ask,
)
from models.model_generator import wrapper_emb_llm, generate_llm_model
from utils.cli import parse_args, validate_args

# Module-level logger; do not configure logging on import so tests can control logging behavior
logger = logging.getLogger(__name__)


def main():
    # Do not configure logging here; keep import-time side effects minimal so tests can import safely

    # parse and validate arguments using the shared CLI utility
    args, parser = parse_args()
    try:
        validate_args(args, parser)
    except SystemExit:
        # argparse may call SystemExit via parser.error(); propagate so CLI shows message
        raise
    except Exception as e:
        logger.error(str(e))
        return 1

    if args.action == "process":
        # Initialize models only when needed
        embeddings_model, llm = wrapper_emb_llm()
        if getattr(args, "model", None):
            llm = generate_llm_model(model_name=args.model)

        logger.info(f"Generating Chroma DB from PDF: {args.pdf_path}")
        generate_chroma_db_from_docs(embeddings_model, args.pdf_path)
        logger.info("Chroma DB generated successfully from PDF.")

    elif args.action == "process-urls":
        # Initialize models only when needed
        embeddings_model, llm = wrapper_emb_llm()
        if getattr(args, "model", None):
            llm = generate_llm_model(model_name=args.model)

        logger.info(f"Generating Chroma DB from {len(args.urls)} URLs")
        try:
            generate_chroma_db_from_urls(
                embeddings_model,
                args.urls,
                max_concurrent=args.max_concurrent,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
            )
            logger.info("Chroma DB generated successfully from URLs.")
        except ValueError as e:
            logger.error(f"Failed to process URLs: {e}")
            return 1

    elif args.action == "process-batch":
        # Initialize models only when needed
        embeddings_model, llm = wrapper_emb_llm()
        if getattr(args, "model", None):
            llm = generate_llm_model(model_name=args.model)

        sources = []

        # Load sources from file if provided
        if args.sources_file:
            try:
                with open(args.sources_file, "r") as f:
                    file_sources = json.load(f)
                    if isinstance(file_sources, list):
                        sources.extend(file_sources)
                    else:
                        logger.error("Sources file must contain a JSON array")
                        return 1
            except (FileNotFoundError, json.JSONDecodeError) as e:
                logger.error(f"Error reading sources file: {e}")
                return 1

        # Add sources from command line
        if args.sources:
            sources.extend(args.sources)

        logger.info(f"Generating Chroma DB from {len(sources)} mixed sources")
        try:
            generate_chroma_db_from_sources(
                embeddings_model,
                sources,
                max_concurrent=args.max_concurrent,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
            )
            logger.info("Chroma DB generated successfully from mixed sources.")
        except ValueError as e:
            logger.error(f"Failed to process sources: {e}")
            return 1

    elif args.action == "query":
        # Initialize models only when needed
        embeddings_model, llm = wrapper_emb_llm()
        if getattr(args, "model", None):
            llm = generate_llm_model(model_name=args.model)

        logger.info("Setting up retriever and QA chain.")
        chain = generate_retriever_chain(embeddings_model, llm, top_k=getattr(args, "top_k", 3))

        answer = ask(args.question, chain=chain)
        logger.info(f"{answer}")

    return 0


if __name__ == "__main__":
    # Configure basic logging only when running as a script to avoid side effects during imports/tests
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    sys.exit(main())
