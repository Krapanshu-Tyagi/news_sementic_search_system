# News Semantic Search

A semantic retrieval API for the 20 Newsgroups text collection. It uses Sentence Transformers to encode documents and queries, FAISS for nearest-neighbour retrieval, and FastAPI to expose search.

## Current implementation

1. Load documents from category folders under data/raw/20_newsgroups.
2. Normalize document text and retain the source category.
3. Generate normalized sentence embeddings with all-MiniLM-L6-v2 by default.
4. Build an exact FAISS inner-product index. With normalized vectors, inner product ranks by cosine similarity.
5. Query the index through the FastAPI POST /search endpoint.

This is a retrieval baseline. Fuzzy clustering and semantic caching are not implemented yet and are not claimed as existing features.

## Setup

Use Python 3.10 or newer. Create and activate a virtual environment, then install dependencies:

    python -m venv .venv
    # Windows PowerShell: .venv\Scripts\Activate.ps1
    # macOS/Linux: source .venv/bin/activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt

## Dataset

The expected dataset layout is data/raw/20_newsgroups/<category>/<document>. The repository currently includes the dataset in this layout. If you replace it, preserve the category subfolders.

## Build the vector index

Run from the repository root:

    python -m scripts.build_index

The command writes generated files to artifacts/: newsgroups.faiss and documents.json. These generated artifacts are ignored by Git.

Set EMBEDDING_MODEL to use another compatible Sentence Transformers model. Set INDEX_DIR to change the generated-artifact directory.

## Start the API

    uvicorn api.main:app --reload

Endpoints:

- GET /health — API and index availability.
- POST /search — semantic search. Example JSON body: {"query": "graphics cards and 3D rendering", "top_k": 5}.
- GET /docs — interactive Swagger documentation.

Search results include document ID, source category, similarity score, and a text excerpt. Similarity scores are ranking signals, not calibrated probabilities.

## Tests

    pytest

## Limitations and roadmap

- Exact FAISS search is suitable for this corpus baseline; approximate indexing can be evaluated if scale requires it.
- Add a labelled relevance benchmark and compare against a TF-IDF baseline before claiming retrieval quality.
- Add model/index version metadata and a safe rebuild process.
- Evaluate a reranker, fuzzy clustering, and semantic caching as separate measurable improvements.
