from .base import BaseEmbedder
from .pretrained import SentenceTransformerEmbedder
from .word2vec import Word2VecEmbedder


def get_embedder(name: str, **kwargs) -> BaseEmbedder:
    """
    Factory function — get an embedder by name.

    Usage:
        embedder = get_embedder("pretrained")
        embedder = get_embedder("pretrained", model_name="all-mpnet-base-v2")
        embedder = get_embedder("word2vec", model_path="models/word2vec.pt")
    """
    registry = {
        "pretrained": SentenceTransformerEmbedder,
        "word2vec": Word2VecEmbedder,
    }
    if name not in registry:
        raise ValueError(f"Unknown embedder '{name}'. Choose from: {list(registry)}")
    return registry[name](**kwargs)


__all__ = ["BaseEmbedder", "SentenceTransformerEmbedder", "Word2VecEmbedder", "get_embedder"]
