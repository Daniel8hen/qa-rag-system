import pytest


@pytest.fixture()
def sample_html():
    """Return a small but sufficient HTML document for extraction tests."""
    return (
        "<html>"
        "<head><title>Example Page Title</title></head>"
        "<body>"
        "<h1>Heading</h1>"
        "<p>" + ("This is sample content. " * 20) + "</p>"
        "</body>"
        "</html>"
    )


# Ensure an event loop fixture is available for pytest-asyncio
@pytest.fixture
def event_loop():
    import asyncio

    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
