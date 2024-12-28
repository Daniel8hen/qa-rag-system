from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

def generate_embedding_model() -> OpenAIEmbeddings:
    """
    Generates an embedding model using the OpenAI API key from the environment.
    :return: Embedding model instance.
    """
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in environment. Check your .env file.")
    return OpenAIEmbeddings(openai_api_key=openai_api_key)

def generate_llm_model(model_name: str = "gpt-4o-mini", max_tokens: int = 200) -> ChatOpenAI:
    """
    Generates an LLM model with specified parameters.
    :param model_name: Name of the LLM model.
    :param max_tokens: Maximum token length for responses.
    :return: Chat model instance.
    """
    return ChatOpenAI(model_name=model_name, max_tokens=max_tokens)

def wrapper_emb_llm():
    """
    Wrapper to generate both embedding and LLM models.
    :return: Tuple containing the embedding model and LLM model.
    """
    return generate_embedding_model(), generate_llm_model()
