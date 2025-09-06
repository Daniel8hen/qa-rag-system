import argparse
import logging
from typing import List, Optional, Tuple


def create_parser() -> argparse.ArgumentParser:
    """Create and return the CLI ArgumentParser for the application.

    The parser uses subcommands to separate actions (process, process-urls,
    process-batch, query) and exposes common processing/querying options.
    """
    parser = argparse.ArgumentParser(
        description="CLI for processing documents from various sources or querying the retriever."
    )

    # Global options
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=5,
        help="Maximum concurrent downloads (default: 5).",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=4000,
        help="Chunk size for text splitting (default: 4000).",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=20,
        help="Chunk overlap for text splitting (default: 20).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to a configuration file or .env (optional).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without persisting any data (useful for testing).",
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="action")

    # process: single PDF
    p_process = subparsers.add_parser("process", help="Process a single PDF file.")
    p_process.add_argument("--pdf_path", type=str, help="Path to PDF file (required).")

    # process-urls: list of URLs
    p_urls = subparsers.add_parser("process-urls", help="Process a list of URLs.")
    p_urls.add_argument("--urls", type=str, nargs="+", help="List of URLs to process (required).")

    # process-batch: mixed sources
    p_batch = subparsers.add_parser("process-batch", help="Process mixed sources (PDFs and URLs).")
    p_batch.add_argument("--sources_file", type=str, help="JSON file containing list of sources.")
    p_batch.add_argument("--sources", type=str, nargs="+", help="List of mixed sources - PDFs and URLs.")

    # query
    p_query = subparsers.add_parser("query", help="Query the existing index.")
    p_query.add_argument("--question", type=str, help="Question to ask (required for query).")
    p_query.add_argument("--top_k", type=int, default=3, help="Number of documents to retrieve (default: 3).")
    p_query.add_argument("--model", type=str, help="Optional model name to override default LLM.")

    return parser


def parse_args(argv: Optional[List[str]] = None) -> Tuple[argparse.Namespace, argparse.ArgumentParser]:
    """Parse CLI arguments and return the (args, parser) tuple.

    Accepts an optional argv list to facilitate testing.
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    # Configure logging level if requested
    if getattr(args, "verbose", False):
        logging.getLogger().setLevel(logging.DEBUG)

    return args, parser


def validate_args(args: argparse.Namespace, parser: Optional[argparse.ArgumentParser] = None) -> None:
    """Perform cross-argument validation and raise a parser error when invalid.

    If parser is provided, use parser.error(...) to produce standard error messaging.
    """
    # convenience local error function
    def _error(msg: str):
        if parser:
            parser.error(msg)
        raise ValueError(msg)

    action = getattr(args, "action", None)
    if not action:
        _error("No action specified. Choose one of: process, process-urls, process-batch, query")

    if action == "process":
        if not getattr(args, "pdf_path", None):
            _error("--pdf_path required for 'process' action.")

    elif action == "process-urls":
        if not getattr(args, "urls", None):
            _error("--urls required for 'process-urls' action.")

    elif action == "process-batch":
        sources_file = getattr(args, "sources_file", None)
        sources = getattr(args, "sources", None)
        if not sources_file and not sources:
            _error("Either --sources_file or --sources required for 'process-batch' action.")

    elif action == "query":
        if not getattr(args, "question", None):
            _error("--question required for 'query' action.")

    # no return; if validation passes we continue
