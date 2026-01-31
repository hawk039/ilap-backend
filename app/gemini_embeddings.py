import os
from typing import List
from google import genai

class GeminiEmbeddingFunction:
    def __init__(self, api_key: str | None = None, model: str = "text-embedding-004"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY missing")
        self.client = genai.Client(api_key=self.api_key)
        self.model = model

    def __call__(self, input: List[str]) -> List[List[float]]:
        # Chroma calls this with a list[str]
        res = self.client.models.embed_content(
            model=self.model,
            contents=input,
        )
        # google-genai returns embeddings aligned with contents
        return [e.values for e in res.embeddings]
