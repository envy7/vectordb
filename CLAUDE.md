# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run examples (pretrained model, no training needed)
uv run python examples/word_clusters.py
uv run python examples/word_clusters.py --interactive
uv run python examples/word_clusters.py --embedder word2vec --model models/word2vec.pt

# Download training corpus (~31MB)
uv run python data/download_corpus.py

# Train Word2Vec from scratch (requires corpus)
uv run train-word2vec
uv run train-word2vec --embed-dim 100 --epochs 5 --corpus data/corpus.txt --output models/word2vec.pt

# Compare both models side by side
uv run python examples/compare_models.py --model models/word2vec.pt
```

## Architecture

**Embedder interface** (`core/embeddings/base.py`): `BaseEmbedder` is the central abstraction. All code that uses embeddings depends only on this interface. `SentenceTransformerEmbedder` and `Word2VecEmbedder` are the two concrete implementations. Add new embedders by subclassing `BaseEmbedder` and implementing `embed()` and `dim`.

**VectorDB** (`core/db.py`): Orchestrates embedder + storage + search. Public API for all insert/query operations. Takes an embedder via constructor injection — this is the single swap point.

**Storage** (`core/storage.py`): Keeps two parallel structures in sync: `_records` (list of `VectorRecord` with metadata) and `_matrix` (numpy array of shape `(n, dim)`). The matrix must always equal `np.vstack([r.vector for r in _records])`. Search always uses `_matrix` — never iterate records for similarity.

**Search** (`core/search.py`): `brute_force_search` does L2-normalise-then-dot-product against the full matrix in one BLAS call. Keep the hot path free of Python loops.

**Training pipeline** (`training/`): Independent of `core/` — imports nothing from it. `corpus.py` → tokenise and build `Vocabulary`. `dataset.py` → `SkipGramDataset` generates `(center, context, negatives)` triples. `trainer.py` → `SkipGramModel` (two `nn.Embedding` layers), trains, saves checkpoint. The checkpoint schema is `{"embeddings": np.ndarray, "word_to_idx": dict, "idx_to_word": dict, "embed_dim": int}` — `Word2VecEmbedder` depends on this exact schema.

**Visualisation** (`viz/`): `reduce.py` wraps sklearn PCA/t-SNE. `plot.py` provides `plot_matplotlib` (static) and `plot_plotly` (interactive). Both accept the same `(coords, labels, groups)` signature.

## Key constraints

- `VectorDB._matrix` and `VectorDB._records` must stay in sync (same order, same length).
- Vectors stored in the matrix and returned by embedders must be `np.float32`.
- `Word2VecEmbedder` handles OOV words by skipping them; an all-OOV input returns a zero vector.
- `perplexity` in t-SNE must be less than `n_samples` — `reduce_tsne` clamps this automatically.
- The training pipeline requires `data/corpus.txt` to exist; run `download_corpus.py` first.
