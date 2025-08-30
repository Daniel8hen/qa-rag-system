import argparse
import logging
import json
from pathlib import Path
from data_interactor import (
    generate_chroma_db_from_docs, 
    generate_chroma_db_from_urls, 
    generate_chroma_db_from_sources,
    generate_retriever_chain, 
    ask
)
from models.model_generator import wrapper_emb_llm

def main():
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Argument parser
    parser = argparse.ArgumentParser(description="CLI for processing documents from various sources or querying the retriever.")
    parser.add_argument("--action", choices=["process", "process-urls", "process-batch", "query"], required=True, 
                       help="Specify the action: process (single PDF), process-urls (URLs), process-batch (mixed sources), or query.")
    
    # Input source arguments
    parser.add_argument("--pdf_path", type=str, help="Path to PDF file (for 'process' action).")
    parser.add_argument("--urls", type=str, nargs="+", help="List of URLs to process (for 'process-urls' action).")
    parser.add_argument("--sources_file", type=str, help="JSON file containing list of sources (for 'process-batch' action).")
    parser.add_argument("--sources", type=str, nargs="+", help="List of mixed sources - PDFs and URLs (for 'process-batch' action).")
    
    # Processing parameters
    parser.add_argument("--max_concurrent", type=int, default=5, help="Maximum concurrent downloads (default: 5).")
    parser.add_argument("--chunk_size", type=int, default=4000, help="Chunk size for text splitting (default: 4000).")
    parser.add_argument("--chunk_overlap", type=int, default=20, help="Chunk overlap for text splitting (default: 20).")
    
    # Query parameters
    parser.add_argument("--question", type=str, help="Question to ask (required for 'query' action).")

    # parsing the arguments from the script
    args = parser.parse_args()

    # Generate embedding and LLM models
    embeddings_model, llm = wrapper_emb_llm()
    logger.info("Embedding and LLM models loaded successfully.")

    if args.action == "process":
        if not args.pdf_path:
            parser.error("--pdf_path required for 'process' action.")

        logger.info(f"Generating Chroma DB from PDF: {args.pdf_path}")
        generate_chroma_db_from_docs(embeddings_model, args.pdf_path)
        logger.info("Chroma DB generated successfully from PDF.")

    elif args.action == "process-urls":
        if not args.urls:
            parser.error("--urls required for 'process-urls' action.")

        logger.info(f"Generating Chroma DB from {len(args.urls)} URLs")
        try:
            generate_chroma_db_from_urls(
                embeddings_model, 
                args.urls, 
                max_concurrent=args.max_concurrent,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap
            )
            logger.info("Chroma DB generated successfully from URLs.")
        except ValueError as e:
            logger.error(f"Failed to process URLs: {e}")
            return 1

    elif args.action == "process-batch":
        sources = []
        
        # Load sources from file if provided
        if args.sources_file:
            try:
                with open(args.sources_file, 'r') as f:
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
        
        if not sources:
            parser.error("Either --sources_file or --sources required for 'process-batch' action.")

        logger.info(f"Generating Chroma DB from {len(sources)} mixed sources")
        try:
            generate_chroma_db_from_sources(
                embeddings_model,
                sources,
                max_concurrent=args.max_concurrent,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap
            )
            logger.info("Chroma DB generated successfully from mixed sources.")
        except ValueError as e:
            logger.error(f"Failed to process sources: {e}")
            return 1

    elif args.action == "query":
        if not args.question:
            parser.error("--question required for 'query' action.")

        logger.info("Setting up retriever and QA chain.")
        chain = generate_retriever_chain(embeddings_model, llm)

        answer = ask(args.question, chain=chain)
        logger.info(f"{answer}")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())