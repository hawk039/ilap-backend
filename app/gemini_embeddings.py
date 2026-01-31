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
        # We must batch internally to respect Gemini limits (e.g. 100 per call)
        BATCH_SIZE = 100
        all_embeddings = []
        
        for i in range(0, len(input), BATCH_SIZE):
            batch = input[i : i + BATCH_SIZE]
            try:
                res = self.client.models.embed_content(
                    model=self.model,
                    contents=batch,
                )
                # google-genai returns embeddings aligned with contents
                # Each embedding object has a .values attribute
                batch_embeddings = [e.values for e in res.embeddings]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"Error embedding batch {i}: {e}")
                # In case of error, we might want to raise or return empty/zeros
                # Raising is safer to avoid silent failures
                raise e

        return all_embeddings

    def name(self) -> str:
        return "GeminiEmbeddingFunction"

    def get_config(self) -> dict:
        return {"model": self.model}
