import pytest
from utils import cli


def test_process_requires_pdf_path():
    # missing pdf_path should raise parser error
    with pytest.raises(SystemExit):
        # argparse triggers SystemExit on error when using parser.parse_args
        cli.parse_args(["process"])[0]


def test_query_requires_question():
    # Using validate_args should raise ValueError when question is missing
    args, parser = cli.parse_args(["query"])
    with pytest.raises(SystemExit):
        # validate_args uses parser.error which calls SystemExit
        cli.validate_args(args, parser)


def test_process_urls_requires_urls():
    args, parser = cli.parse_args(["process-urls"])
    with pytest.raises(SystemExit):
        cli.validate_args(args, parser)


def test_process_batch_requires_sources():
    args, parser = cli.parse_args(["process-batch"])
    with pytest.raises(SystemExit):
        cli.validate_args(args, parser)


def test_parse_valid_query_args():
    args, parser = cli.parse_args(["query", "--question", "What is RAG?", "--top_k", "2"])
    # Should not raise
    cli.validate_args(args, parser)
    assert args.question == "What is RAG?"
    assert args.top_k == 2
