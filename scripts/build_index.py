import json
import os
from pathlib import Path

from data.dataloader import load_data
from embeddings.embedder import Embedder
from vector_db.faiss_store import build_index, save_index

ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = Path(os.getenv("INDEX_DIR", ROOT / "artifacts"))


def main() -> None:
    documents, labels = load_data()
    print(f"Loaded {len(documents):,} documents across {len(set(labels))} categories.")

    embeddings = Embedder().embed(documents)
    print(f"Generated embeddings with shape {embeddings.shape}.")

    index = build_index(embeddings)
    save_index(index, INDEX_DIR / "newsgroups.faiss")

    metadata = [
        {"document_id": i, "category": labels[i], "text": documents[i]}
        for i in range(len(documents))
    ]
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    (INDEX_DIR / "documents.json").write_text(
        json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved {index.ntotal:,} vectors and document metadata to {INDEX_DIR}.")


if __name__ == "__main__":
    main()
