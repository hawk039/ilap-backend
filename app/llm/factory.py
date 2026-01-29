import os
from .base import BaseLLM
from .local_llm import LocalLLM

def get_llm() -> BaseLLM:
    provider = os.getenv("LLM_PROVIDER", "local").lower()

    if provider == "gemini":
        from .gemini_llm import GeminiLLM
        return GeminiLLM()

    # safe default
    return LocalLLM()
