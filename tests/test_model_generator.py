import pytest
from unittest.mock import patch, MagicMock
from src.models.model_generator import (
    generate_embedding_model,
    generate_llm_model,
    wrapper_emb_llm,
)
import sys
import os

print(sys.path)
# Add the project root or 'src' folder to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))


@patch("src.models.model_generator.os.getenv")
@patch("src.models.model_generator.OpenAIEmbeddings")
def test_generate_embedding_model(mock_embeddings, mock_getenv):
    # Mock the environment variable
    mock_getenv.return_value = "fake_api_key"
    # Mock the OpenAIEmbeddings instantiation
    mock_embeddings.return_value = MagicMock()

    # Call the function
    embedding_model = generate_embedding_model()

    # Assertions
    mock_getenv.assert_called_once_with("OPENAI_API_KEY")
    mock_embeddings.assert_called_once_with(openai_api_key="fake_api_key")
    assert embedding_model == mock_embeddings.return_value


@patch("src.models.model_generator.ChatOpenAI")
def test_generate_llm_model(mock_chat_openai):
    # Mock the ChatOpenAI instantiation
    mock_chat_openai.return_value = MagicMock()

    # Call the function with parameters
    llm_model = generate_llm_model(model_name="gpt-4o-mini", max_tokens=200)

    # Assertions
    mock_chat_openai.assert_called_once_with(model_name="gpt-4o-mini", max_tokens=200)
    assert llm_model == mock_chat_openai.return_value


@patch("src.models.model_generator.generate_embedding_model")
@patch("src.models.model_generator.generate_llm_model")
def test_wrapper_emb_llm(mock_generate_llm, mock_generate_embedding):
    # Mock the individual functions
    mock_generate_embedding.return_value = "mock_embedding_model"
    mock_generate_llm.return_value = "mock_llm_model"

    # Call the wrapper function
    embedding_model, llm_model = wrapper_emb_llm()

    # Assertions
    mock_generate_embedding.assert_called_once()
    mock_generate_llm.assert_called_once()
    assert embedding_model == "mock_embedding_model"
    assert llm_model == "mock_llm_model"
