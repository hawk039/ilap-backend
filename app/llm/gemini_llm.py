import os
from .base import BaseLLM

class GeminiLLM(BaseLLM):
    def __init__(self):
        from google import genai  # lazy import (keeps app boot fast/fail-safe)
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")

        self.client = genai.Client(api_key=api_key)
        self.model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

    def generate(self, prompt: str) -> str:
        resp = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )
        # defensive: some SDK responses can be None
        text = getattr(resp, "text", None)
        return (text or "").strip()
