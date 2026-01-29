from .base import BaseLLM

class LocalLLM(BaseLLM):
    def generate(self, prompt: str) -> str:
        # Temporary deterministic stub
        return "Based on the retrieved legal provisions, the applicable law is as follows."
