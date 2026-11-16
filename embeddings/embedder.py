import os
from sentence_transformers import SentenceTransformer


class Embedder:
    """Sentence-transformer wrapper returning normalized dense vectors."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.model = SentenceTransformer(self.model_name)

    def embed(self, texts: list[str]):
        if not texts:
            raise ValueError("Cannot embed an empty text list.")
        return self.model.encode(
            texts,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 128,
            convert_to_numpy=True,
        )
