import numpy as np
import pytest

faiss = pytest.importorskip("faiss")

from vector_db.faiss_store import build_index, search_index


def test_search_returns_nearest_normalized_vector():
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    index = build_index(vectors)

    scores, indices = search_index(index, np.array([0.9, 0.1]), top_k=1)

    assert indices == [0]
    assert scores[0] > 0.8


def test_build_index_rejects_empty_embeddings():
    with pytest.raises(ValueError):
        build_index(np.empty((0, 3), dtype="float32"))
