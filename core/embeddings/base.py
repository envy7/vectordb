"""
base.py — Abstract interface that all embedding models must implement.

Why an abstract base class?
    We want to swap Word2Vec ↔ sentence-transformers without changing any
    downstream code (VectorDB, search, visualization). The BaseEmbedder
    contract is the only thing those modules depend on.

    This is the Dependency Inversion Principle: high-level modules (VectorDB)
    depend on the abstraction (BaseEmbedder), not the concrete implementation.
"""

from abc import ABC, abstractmethod

import numpy as np


class BaseEmbedder(ABC):
    """
    All embedding models must subclass this and implement `embed` and `dim`.

    An embedding model converts raw text → a dense float vector.
    The vector lives in a high-dimensional space where semantically similar
    texts end up geometrically close (small cosine distance).
    """

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """
        Convert a single string to a 1-D float32 numpy array of shape (dim,).

        The returned vector should be consistent: the same text should always
        produce the same vector (deterministic after model load).
        """
        ...

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """
        Convert multiple strings to a 2-D array of shape (n, dim).

        Default implementation: loop over embed(). Subclasses can override
        this for batched inference (sentence-transformers does this efficiently).
        """
        return np.array([self.embed(t) for t in texts], dtype=np.float32)

    @property
    @abstractmethod
    def dim(self) -> int:
        """
        Dimensionality of the output vectors.

        This varies by model:
            - Our Word2Vec: whatever embed_dim we trained with (e.g. 100)
            - all-MiniLM-L6-v2: 384
            - all-mpnet-base-v2: 768
            - OpenAI text-embedding-3-small: 1536
        """
        ...

    @property
    def name(self) -> str:
        """Human-readable name shown in logs and visualisation titles."""
        return self.__class__.__name__
