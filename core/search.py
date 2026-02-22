"""
search.py — Similarity search algorithms.

The core question: "Given a query vector, which stored vectors are most similar?"

Similarity metric: Cosine Similarity
    cos(A, B) = (A · B) / (||A|| * ||B||)

    Range: [-1, 1]  where 1 = identical direction, 0 = orthogonal, -1 = opposite
    We use cosine (angle) rather than Euclidean (distance) because:
    - Embedding magnitudes don't carry meaning — only direction does
    - "Paris" and "Paris is a city" may have different magnitudes but similar direction
    - Cosine is invariant to vector scale

    After normalizing both vectors to unit length, cosine similarity reduces to
    just a dot product: cos(A, B) = A_norm · B_norm
    This lets us compute similarities for ALL stored vectors in one matrix multiply.

Two search modes are implemented here:

1. brute_force_search — exact, O(n·d)
   Compares query against every vector. Always finds the true top-k.
   Fast enough up to ~100k vectors.

2. (Future) HNSW — approximate, O(log n)
   Hierarchical Navigable Small World graph. Used by Qdrant, Weaviate, FAISS.
   Trades a tiny bit of recall for massive speed gains at scale.

Further reading:
    Cosine similarity: https://en.wikipedia.org/wiki/Cosine_similarity
    ANN benchmarks:    https://ann-benchmarks.com/
    HNSW paper:        https://arxiv.org/abs/1603.09320
"""

from dataclasses import dataclass

import numpy as np

from .storage import VectorRecord, VectorStorage


@dataclass
class SearchResult:
    """A single search hit returned by a query."""
    record: VectorRecord
    score: float    # cosine similarity in [-1, 1]; higher = more similar
    rank: int       # 1-based rank (1 = most similar)


def _cosine_similarities(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between `query` (1-D) and every row of `matrix`.

    Returns a 1-D array of shape (n,) with similarity scores.

    Steps:
        1. L2-normalize query to unit length
        2. L2-normalize each row of matrix to unit length
        3. Dot product (matrix @ query) = cosine similarity for unit vectors

    The +1e-10 epsilon prevents division-by-zero for zero vectors.
    """
    # Normalize query: shape (dim,) → unit vector
    query_norm = query / (np.linalg.norm(query) + 1e-10)

    # Normalize each row of matrix: shape (n, dim) → each row is unit length
    row_norms = np.linalg.norm(matrix, axis=1, keepdims=True)  # (n, 1)
    matrix_norm = matrix / (row_norms + 1e-10)                 # (n, dim)

    # Matrix-vector multiply: (n, dim) @ (dim,) = (n,)
    # This is one BLAS call — very fast even for large n
    return matrix_norm @ query_norm


def brute_force_search(
    query_vector: np.ndarray,
    storage: VectorStorage,
    top_k: int = 5,
) -> list[SearchResult]:
    """
    Exact nearest-neighbour search by linear scan.

    Every stored vector is scored. Returns the top_k most similar records
    sorted by descending cosine similarity.

    Time complexity: O(n * d) per query
        n = number of stored vectors
        d = embedding dimension

    This is called "flat search" in FAISS terminology (IndexFlatIP).
    """
    matrix = storage.get_matrix()
    if matrix is None or len(storage) == 0:
        return []

    # Score all vectors at once with one matrix multiply
    scores = _cosine_similarities(query_vector, matrix)

    # argsort returns indices in ascending order → take last top_k → reverse
    top_k = min(top_k, len(scores))
    top_indices = np.argsort(scores)[-top_k:][::-1]

    records = storage.get_records()
    return [
        SearchResult(
            record=records[i],
            score=float(scores[i]),
            rank=rank + 1,
        )
        for rank, i in enumerate(top_indices)
    ]
