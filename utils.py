import os
from pydantic import SecretStr
from dotenv import load_dotenv

load_dotenv()


def get_openrouter_base_url() -> str:
    """Retrieve the OpenRouter base URL from environment variables."""
    base_url = os.environ.get("OPENROUTER_BASE_URL")
    if not base_url:
        raise ValueError("OPENROUTER_BASE_URL environment variable is not set.")
    return base_url

def get_openrouter_api_key() -> SecretStr:
    """Retrieve the OpenRouter API key from environment variables."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is not set.")
    return SecretStr(api_key)