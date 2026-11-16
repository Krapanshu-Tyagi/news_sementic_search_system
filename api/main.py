import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from embeddings.embedder import Embedder
from vector_db.faiss_store import load_index, search_index

ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = Path(os.getenv("INDEX_DIR", ROOT / "artifacts"))
INDEX_PATH = INDEX_DIR / "newsgroups.faiss"
DOCUMENTS_PATH = INDEX_DIR / "documents.json"
_resources: dict[str, Any] = {}

app = FastAPI(
    title="News Semantic Search API",
    description="Semantic retrieval over the 20 Newsgroups text collection.",
    version="1.0.0",
)


class SearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)


def get_resources():
    if _resources:
        return _resources
    if not INDEX_PATH.is_file() or not DOCUMENTS_PATH.is_file():
        raise HTTPException(
            status_code=503,
            detail="Search index is not built. Run python -m scripts.build_index first.",
        )
    try:
        _resources.update(
            index=load_index(INDEX_PATH),
            documents=json.loads(DOCUMENTS_PATH.read_text(encoding="utf-8")),
            embedder=Embedder(),
        )
    except Exception as exc:
        _resources.clear()
        raise HTTPException(status_code=503, detail=f"Could not load search resources: {exc}") from exc
    return _resources


@app.get("/health")
def health():
    return {
        "status": "ok",
        "index_available": INDEX_PATH.is_file() and DOCUMENTS_PATH.is_file(),
    }


@app.post("/search")
def search(request: SearchRequest):
    resources = get_resources()
    try:
        query_vector = resources["embedder"].embed([request.query])[0]
        scores, indices = search_index(resources["index"], query_vector, request.top_k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}") from exc

    results = []
    for score, idx in zip(scores, indices):
        if idx < 0:
            continue
        item = resources["documents"][idx]
        results.append({
            "document_id": item["document_id"],
            "category": item["category"],
            "score": float(score),
            "text": item["text"][:1200],
        })
    return {"query": request.query, "count": len(results), "results": results}
