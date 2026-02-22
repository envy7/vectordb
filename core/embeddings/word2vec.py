"""
word2vec.py — Inference wrapper for our from-scratch Word2Vec model.

This module handles LOADING and USING a trained model. Training lives in
training/trainer.py — keep them separate so you can import this without
pulling in PyTorch training dependencies.

How Word2Vec embeddings work (inference):
    1. Split text into words (simple whitespace split)
    2. Look up each word's vector in the embedding matrix
    3. Average all word vectors → one vector for the whole text

    This averaging is called "bag-of-words" because word order is ignored.
    "dog bites man" and "man bites dog" produce the same vector.
    That's a limitation — but for many retrieval tasks it works surprisingly well.

    sentence-transformers uses a transformer to capture word order and context,
    which is why it handles more complex semantics better.
"""

import numpy as np
import torch

from .base import BaseEmbedder


class Word2VecEmbedder(BaseEmbedder):
    """
    Loads a Word2Vec model saved by training/trainer.py and uses it to embed text.

    The saved checkpoint contains:
        embeddings   : np.ndarray of shape (vocab_size, embed_dim)
        word_to_idx  : dict mapping word → integer index
        idx_to_word  : dict mapping integer index → word
        embed_dim    : int
    """

    def __init__(self, model_path: str):
        """
        Args:
            model_path: Path to the .pt file saved by training/trainer.py
                        e.g. "models/word2vec.pt"
        """
        print(f"Loading Word2Vec model from: {model_path}")
        # weights_only=True is safer — avoids arbitrary code execution from pickle
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        # The embedding matrix: rows = words, columns = embedding dimensions
        self._embeddings: np.ndarray = checkpoint["embeddings"]  # (vocab_size, embed_dim)
        self._word_to_idx: dict[str, int] = checkpoint["word_to_idx"]
        self._dim: int = self._embeddings.shape[1]

        print(f"  → Vocab size: {len(self._word_to_idx):,} | Embedding dim: {self._dim}")

    def embed(self, text: str) -> np.ndarray:
        """
        Embed text by averaging its word vectors.

        Words not in the vocabulary (OOV — out-of-vocabulary) are silently
        skipped. If NO words are in the vocab (e.g. pure numbers), returns
        a zero vector.
        """
        words = text.lower().split()
        vectors = []

        for word in words:
            if word in self._word_to_idx:
                idx = self._word_to_idx[word]
                vectors.append(self._embeddings[idx])
            # OOV words are skipped — could also use a special <UNK> vector

        if not vectors:
            # All words unknown → return zero vector
            # Zero vectors will have undefined cosine similarity — be aware
            return np.zeros(self._dim, dtype=np.float32)

        # np.mean across axis=0 averages element-wise across all word vectors
        return np.mean(vectors, axis=0).astype(np.float32)

    def has_word(self, word: str) -> bool:
        """Check if a specific word is in the trained vocabulary."""
        return word.lower() in self._word_to_idx

    def get_word_vector(self, word: str) -> np.ndarray | None:
        """Get the raw embedding for a single word (not averaged)."""
        word = word.lower()
        if word not in self._word_to_idx:
            return None
        return self._embeddings[self._word_to_idx[word]]

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def vocab_size(self) -> int:
        return len(self._word_to_idx)

    @property
    def name(self) -> str:
        return f"Word2Vec(dim={self._dim}, vocab={self.vocab_size:,})"
