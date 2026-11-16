from pathlib import Path
import faiss
import numpy as np


def build_index(embeddings: np.ndarray):
    vectors = np.asarray(embeddings, dtype="float32")
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise ValueError("Embeddings must be a non-empty 2D array.")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def save_index(index, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(destination))


def load_index(path: str | Path):
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"FAISS index not found: {source}")
    return faiss.read_index(str(source))


def search_index(index, query_embedding: np.ndarray, top_k: int):
    vector = np.asarray(query_embedding, dtype="float32").reshape(1, -1)
    if vector.shape[1] != index.d:
        raise ValueError("Query embedding dimension does not match the index.")
    scores, indices = index.search(vector, min(top_k, index.ntotal))
    return scores[0].tolist(), indices[0].tolist()
