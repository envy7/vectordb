"""
storage.py — In-memory vector store with optional file persistence.

Why store vectors in a matrix instead of a list of arrays?
    Matrix layout: all vectors packed into one contiguous numpy array of
    shape (n, dim). This is critical for search performance.

    When searching, we need to compare the query against EVERY stored vector.
    With a matrix, that's a single matrix-vector multiplication (one BLAS call).
    With a list of arrays, we'd need n separate dot products — much slower
    and worse for CPU cache (data isn't contiguous in memory).

Persistence design:
    We split storage into two files:
    - vectors.npy  : binary numpy file — compact, fast load (mmap-able)
    - metadata.json: human-readable — IDs, original text, extra fields

    Real vector DBs (Qdrant, Milvus) use more sophisticated formats like
    memory-mapped files, columnar storage (Lance/Arrow), or custom segment files.
    But this two-file approach captures the core idea.
"""

import json
import os
from dataclasses import dataclass, field

import numpy as np


@dataclass
class VectorRecord:
    """One stored item: its vector plus all associated metadata."""
    id: str
    vector: np.ndarray      # shape: (dim,)
    text: str               # original text before embedding
    metadata: dict = field(default_factory=dict)  # arbitrary user-provided fields


class VectorStorage:
    """
    In-memory store that keeps all vectors in a stacked numpy matrix.

    Internally maintains two parallel structures:
        _records : list of VectorRecord  (metadata, text, id)
        _matrix  : np.ndarray (n, dim)   (all vectors stacked row-by-row)

    They must stay in sync: _matrix[i] is always _records[i].vector.
    """

    def __init__(self):
        self._records: list[VectorRecord] = []
        # None until the first vector is added (we don't know dim yet)
        self._matrix: np.ndarray | None = None

    def add(self, record: VectorRecord) -> None:
        """Append a record and extend the vector matrix."""
        self._records.append(record)
        vec = record.vector.reshape(1, -1)  # (dim,) → (1, dim) for vstack

        if self._matrix is None:
            self._matrix = vec
        else:
            # vstack creates a new array each time — O(n) copy.
            # For high-throughput inserts, a real DB would use pre-allocated
            # buffers or append-only segment files.
            self._matrix = np.vstack([self._matrix, vec])

    def get_matrix(self) -> np.ndarray | None:
        """Return the full (n, dim) matrix of all stored vectors."""
        return self._matrix

    def get_records(self) -> list[VectorRecord]:
        return self._records

    def get_by_id(self, record_id: str) -> VectorRecord | None:
        """Linear scan by ID — fine for small collections."""
        for r in self._records:
            if r.id == record_id:
                return r
        return None

    def __len__(self) -> int:
        return len(self._records)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str) -> None:
        """
        Persist the store to disk.

        Creates:
            <directory>/vectors.npy    — the vector matrix (binary, compact)
            <directory>/metadata.json  — IDs, texts, metadata (human-readable)
        """
        os.makedirs(directory, exist_ok=True)

        if self._matrix is not None:
            np.save(os.path.join(directory, "vectors.npy"), self._matrix)

        meta = [
            {"id": r.id, "text": r.text, "metadata": r.metadata}
            for r in self._records
        ]
        with open(os.path.join(directory, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

        print(f"Saved {len(self)} vectors to {directory}/")

    @classmethod
    def load(cls, directory: str) -> "VectorStorage":
        """Load a previously saved store from disk."""
        storage = cls()

        matrix = np.load(os.path.join(directory, "vectors.npy"))
        with open(os.path.join(directory, "metadata.json")) as f:
            meta = json.load(f)

        for i, m in enumerate(meta):
            storage._records.append(VectorRecord(
                id=m["id"],
                vector=matrix[i],
                text=m["text"],
                metadata=m.get("metadata", {}),
            ))
        storage._matrix = matrix

        print(f"Loaded {len(storage)} vectors from {directory}/")
        return storage
