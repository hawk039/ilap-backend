import os
from dotenv import load_dotenv
from .base import BaseLLM
from .local_llm import LocalLLM

# Load environment variables from .env file
load_dotenv()

def get_llm() -> BaseLLM:
    provider = os.getenv("LLM_PROVIDER", "local").lower()

    if provider == "gemini":
        from .gemini_llm import GeminiLLM
        return GeminiLLM()

    # safe default
    return LocalLLM()
