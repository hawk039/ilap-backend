import os
from typing import List, Dict, Any
from google import genai

class GeminiEmbeddingFunction:
    def __init__(self, api_key: str | None = None, model: str = "text-embedding-004"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY missing")
        self.client = genai.Client(api_key=self.api_key)
        self.model = model

    # ✅ Chroma expects this
    def name(self) -> str:
        return f"gemini::{self.model}"

    # ✅ Some Chroma versions expect this too
    def get_config(self) -> Dict[str, Any]:
        return {"model": self.model}

    # Chroma calls this with list[str]
    def __call__(self, input: List[str]) -> List[List[float]]:
        res = self.client.models.embed_content(
            model=self.model,
            contents=input,
        )
        return [e.values for e in res.embeddings]