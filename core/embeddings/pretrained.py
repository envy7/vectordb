"""
pretrained.py — Wrapper around the sentence-transformers library.

sentence-transformers trains models using *contrastive learning*:
    pairs of semantically similar sentences are pulled together in vector
    space, while dissimilar pairs are pushed apart. This is done at massive
    scale (billions of sentence pairs from the web, NLI datasets, etc.).

    The result is a model that produces *sentence-level* embeddings — not
    just word embeddings. "The bank by the river" and "The riverbank" will
    map to nearby vectors, even though they share no words.

Useful models (all free, run locally):
    all-MiniLM-L6-v2   → 384 dims, fast, great quality  (recommended start)
    all-mpnet-base-v2  → 768 dims, slower, better quality
    paraphrase-MiniLM-L3-v2 → 384 dims, even faster (tiny model)

More info: https://www.sbert.net/docs/sentence_transformer/pretrained_models.html
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from .base import BaseEmbedder


class SentenceTransformerEmbedder(BaseEmbedder):
    """
    Production-grade embedder backed by a pre-trained transformer model.

    On first use, the model weights (~90 MB for MiniLM) are downloaded from
    Hugging Face and cached locally in ~/.cache/huggingface/.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Args:
            model_name: Any model from https://huggingface.co/sentence-transformers
        """
        print(f"Loading sentence-transformers model: {model_name}")
        self._model = SentenceTransformer(model_name)
        # get_sentence_embedding_dimension() reads the model config
        self._dim = self._model.get_sentence_embedding_dimension()
        self._model_name = model_name
        print(f"  → Loaded. Embedding dim: {self._dim}")

    def embed(self, text: str) -> np.ndarray:
        # convert_to_numpy=True returns a numpy array instead of a tensor
        return self._model.encode(text, convert_to_numpy=True).astype(np.float32)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """
        sentence-transformers handles batching natively — much faster than
        calling embed() in a loop because it processes multiple texts through
        the transformer in one forward pass.
        """
        return self._model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 50,
        ).astype(np.float32)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return f"SentenceTransformer({self._model_name})"
