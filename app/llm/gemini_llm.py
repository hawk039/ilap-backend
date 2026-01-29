# app/llm/gemini_llm.py
import os
from .base import BaseLLM

class GeminiLLM(BaseLLM):
    def __init__(self) -> None:
        from google import genai  # ✅ NEW SDK: google-genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")

        self.client = genai.Client(api_key=api_key)
        self.model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    def generate(self, prompt: str) -> str:
        resp = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )
        return (getattr(resp, "text", "") or "").strip()
