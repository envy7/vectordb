"""
db.py — The main VectorDB class. Public API.

This is the "orchestrator" that wires together:
    Embedder  →  Storage  →  Search

Users of this library only need to interact with VectorDB.
They should never need to import storage.py or search.py directly.

Design principle: the embedder is injected at construction time (dependency
injection). Swapping the embedding model is a one-line change:

    db = VectorDB(embedder=Word2VecEmbedder("models/word2vec.pt"))
    db = VectorDB(embedder=SentenceTransformerEmbedder("all-MiniLM-L6-v2"))

Everything else (add, search, save, load) stays identical.
"""

from .embeddings.base import BaseEmbedder
from .search import SearchResult, brute_force_search
from .storage import VectorRecord, VectorStorage


class VectorDB:
    """
    A simple vector database.

    Usage:
        from core import VectorDB, get_embedder

        db = VectorDB(embedder=get_embedder("pretrained"))

        db.add("doc1", "The quick brown fox")
        db.add("doc2", "A fast auburn canine")
        db.add("doc3", "Python is a programming language")

        results = db.search("speedy fox", top_k=2)
        for r in results:
            print(f"[{r.score:.3f}] {r.record.text}")
    """

    def __init__(self, embedder: BaseEmbedder):
        self.embedder = embedder
        self.storage = VectorStorage()

    # ------------------------------------------------------------------
    # Inserting data
    # ------------------------------------------------------------------

    def add(self, id: str, text: str, metadata: dict | None = None) -> None:
        """
        Embed `text` and store the resulting vector with the given `id`.

        Args:
            id:       Unique identifier for this item (e.g. "doc_42")
            text:     Raw text to embed and store
            metadata: Optional dict of extra fields (e.g. {"source": "wiki"})
        """
        vector = self.embedder.embed(text)
        record = VectorRecord(
            id=id,
            vector=vector,
            text=text,
            metadata=metadata or {},
        )
        self.storage.add(record)

    def add_batch(self, items: list[dict]) -> None:
        """
        Add multiple items in one call. Faster than looping over add() because
        embed_batch() can process all texts in a single forward pass.

        Args:
            items: list of dicts, each with keys:
                   - "id"       (required)
                   - "text"     (required)
                   - "metadata" (optional)

        Example:
            db.add_batch([
                {"id": "a", "text": "cat"},
                {"id": "b", "text": "dog", "metadata": {"type": "animal"}},
            ])
        """
        texts = [item["text"] for item in items]
        vectors = self.embedder.embed_batch(texts)  # (n, dim) in one shot

        for item, vector in zip(items, vectors):
            record = VectorRecord(
                id=item["id"],
                vector=vector,
                text=item["text"],
                metadata=item.get("metadata", {}),
            )
            self.storage.add(record)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """
        Find the top_k most similar items to the query text.

        The query goes through the same embedder as the stored items —
        this is important. You can't mix vectors from different models.

        Args:
            query:  Natural language query (or a word, phrase, sentence)
            top_k:  How many results to return

        Returns:
            List of SearchResult objects sorted by descending cosine similarity.
        """
        query_vector = self.embedder.embed(query)
        return brute_force_search(query_vector, self.storage, top_k)

    def search_by_vector(self, vector, top_k: int = 5) -> list[SearchResult]:
        """
        Search using a raw vector instead of text.
        Useful when you already have an embedding (e.g. from another model).
        """
        import numpy as np
        return brute_force_search(np.array(vector, dtype=np.float32), self.storage, top_k)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str) -> None:
        """Save all vectors and metadata to disk."""
        self.storage.save(directory)

    def load(self, directory: str) -> None:
        """Load vectors and metadata from a previously saved directory."""
        self.storage = VectorStorage.load(directory)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.storage)

    def __repr__(self) -> str:
        return (
            f"VectorDB("
            f"embedder={self.embedder.name}, "
            f"vectors={len(self)}, "
            f"dim={self.embedder.dim}"
            f")"
        )
