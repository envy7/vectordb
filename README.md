# vectordb — A Vector Database From Scratch

A learning project that builds a vector database from the ground up, including a Word2Vec embedding model trained on raw text. The goal is to understand how embeddings, similarity search, and vector storage actually work — not just use them as black boxes.

---

## What's in here?

| Component | What it teaches |
|---|---|
| `core/embeddings/` | What an embedding model interface looks like; how to swap models |
| `core/storage.py` | Why vectors are stored as a matrix, not a list |
| `core/search.py` | How cosine similarity search works mathematically |
| `training/` | How Word2Vec learns word relationships from raw text |
| `viz/` | How to see embeddings (t-SNE vs PCA) |
| `examples/` | End-to-end demos |

---

## Theory

### What is an embedding?

An embedding maps discrete things (words, sentences, images) to continuous vectors in $\mathbb{R}^d$. The mapping is learned so that **semantically similar things end up geometrically close**.

```
"cat"  → [0.21, -0.87, 0.43, ...]   # 100-384 dimensional vector
"dog"  → [0.19, -0.91, 0.51, ...]   # close to "cat"
"pizza"→ [-0.32, 0.55, -0.11, ...]  # far from "cat"
```

The vector itself has no human-interpretable meaning per dimension. The only thing that matters is **relative distance**.

**Further reading:**
- [The Illustrated Word2Vec](https://jalammar.github.io/illustrated-word2vec/) — best visual explanation
- [Embeddings: What they are and why they matter](https://simonwillison.net/2023/Oct/23/embeddings/)

### How does Word2Vec work?

Word2Vec (Skip-gram variant) trains a neural network to predict context words from a center word:

```
"The quick brown fox jumps"
         ↑
     center word

Predict: "quick", "fox" (within window_size=2)
```

The network is a lookup table (embedding matrix) → dot product → sigmoid. After training on millions of (center, context) pairs, words that appear in similar contexts end up with similar vectors.

The famous result: `king - man + woman ≈ queen` — arithmetic in embedding space!

**Further reading:**
- [Original paper (Mikolov et al. 2013)](https://arxiv.org/abs/1301.3781)
- [Negative sampling explained](https://arxiv.org/abs/1402.3722)

### How does similarity search work?

We use **cosine similarity**: the cosine of the angle between two vectors.

$$\text{cos}(A, B) = \frac{A \cdot B}{\|A\| \cdot \|B\|}$$

Range: `[-1, 1]`. Higher = more similar. We use cosine (angle) rather than Euclidean distance because embedding magnitudes don't carry meaning — only direction does.

After normalising vectors to unit length, cosine similarity = dot product. This lets us compare one query against all stored vectors with a **single matrix multiplication** — the key to efficient search.

**Further reading:**
- [Cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity)
- [ANN benchmarks](https://ann-benchmarks.com/) — how production systems compare

### Why t-SNE for visualisation?

We can't plot 384-dimensional vectors directly, so we reduce to 2D. Two options:

| Method | Type | Speed | Quality |
|---|---|---|---|
| PCA | Linear | Fast | Shows global structure; misses curves |
| t-SNE | Non-linear | Slow | Reveals local clusters much better |

t-SNE works by preserving **local neighbourhoods**: if words are close in 384D, they should be close in 2D. It's non-linear so it can capture the curved manifold that embeddings live on.

**Further reading:**
- [How to Use t-SNE Effectively](https://distill.pub/2016/misread-tsne/) ← must-read interactive article
- [t-SNE paper (van der Maaten & Hinton)](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf)

---

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies
uv sync
```

---

## Quickstart — 5 minutes to a working vector DB

```bash
# 1. Run the word clusters visualisation (uses pretrained model, downloads ~90MB on first run)
uv run python examples/word_clusters.py

# 2. Interactive browser chart
uv run python examples/word_clusters.py --interactive

# 3. Save the plot to a file
uv run python examples/word_clusters.py --save output.png
```

You should see five colour-coded clusters: animals, countries, royalty, technology, food.

---

## CLI — add, search, and visualize from the terminal

The `vdb` command lets you build and query a database without writing any Python. The database persists to disk (`.vdb/` by default) between calls, so you can build it up incrementally.

### Add words

```bash
# Add individual words or multi-word phrases
uv run vdb add cat dog wolf lion tiger bear --group animals
uv run vdb add france germany japan italy --group countries
uv run vdb add "machine learning" "neural network" "deep learning" --group ai

# --group is optional; it controls the colour in viz and appears in search results
uv run vdb add sunrise sunset twilight
```

### Search

```bash
# Find the 5 nearest neighbors (default)
uv run vdb search "big cat"

# Return more results
uv run vdb search "artificial intelligence" --top-k 10
```

Example output:

```
         Nearest to "big cat"
┏━━━━━━━━┳━━━━━━━┳━━━━━━━━┳━━━━━━━━━┓
┃ Rank   ┃ Word  ┃ Score  ┃ Group   ┃
┡━━━━━━━━╇━━━━━━━╇━━━━━━━━╇━━━━━━━━━┩
│ 1      │ cat   │ 0.7745 │ animals │
│ 2      │ dog   │ 0.4805 │ animals │
│ 3      │ tiger │ 0.4721 │ animals │
│ 4      │ lion  │ 0.4711 │ animals │
│ 5      │ bear  │ 0.3958 │ animals │
└────────┴───────┴────────┴─────────┘
```

### List all stored words

```bash
uv run vdb list
```

### Visualize

```bash
# Static matplotlib plot (opens a window)
uv run vdb viz

# Interactive Plotly chart in the browser (hover, zoom, toggle groups)
uv run vdb viz --interactive

# Save to a PNG file instead of displaying
uv run vdb viz --save plot.png

# Use PCA instead of t-SNE (faster, better for small word sets)
uv run vdb viz --method pca
```

### Clear the database

```bash
uv run vdb clear
```

### Options

```
--db <path>        Database directory (default: .vdb)
--embedder         pretrained | word2vec  — chosen once at DB creation (default: pretrained)
--model <path>     Path to Word2Vec checkpoint (when using --embedder word2vec)
```

To maintain multiple separate databases, use `--db`:

```bash
uv run vdb --db animals.vdb add cat dog wolf --group animals
uv run vdb --db places.vdb add paris london tokyo --group cities
```

---

## Training Word2Vec from scratch

### Step 1 — Download the corpus

```bash
# Downloads the text8 Wikipedia dataset (~31MB), takes the first 1M tokens
uv run python data/download_corpus.py

# Or take more tokens for better quality (slower training)
uv run python data/download_corpus.py --max-tokens 5000000
```

### Step 2 — Train

```bash
# Default: 100-dim embeddings, 5 epochs, ~5 minutes on CPU
uv run train-word2vec

# Or with custom settings
uv run train-word2vec \
    --corpus data/corpus.txt \
    --output models/word2vec.pt \
    --embed-dim 100 \
    --epochs 5 \
    --window 2 \
    --min-count 5
```

### Step 3 — Visualise and compare

```bash
# Visualise clusters with your trained model
uv run python examples/word_clusters.py --embedder word2vec --model models/word2vec.pt

# Side-by-side comparison: your model vs pretrained
uv run python examples/compare_models.py --model models/word2vec.pt
```

---

## Embedding and search

The core functionality — embed text and find the closest matches — works out of the box with no training required.

### Searching with natural language

You don't have to search by exact word. The query goes through the same embedder as the stored items, so you can search using descriptions and the model understands meaning:

```python
from core import VectorDB, get_embedder

db = VectorDB(embedder=get_embedder("pretrained"))

db.add_batch([
    {"id": "cat",     "text": "cat"},
    {"id": "dog",     "text": "dog"},
    {"id": "lion",    "text": "lion"},
    {"id": "tiger",   "text": "tiger"},
    {"id": "king",    "text": "king"},
    {"id": "queen",   "text": "queen"},
    {"id": "france",  "text": "france"},
    {"id": "germany", "text": "germany"},
    {"id": "pizza",   "text": "pizza"},
    {"id": "pasta",   "text": "pasta"},
    {"id": "computer","text": "computer"},
    {"id": "software","text": "software"},
])

for r in db.search("feline animal", top_k=3):
    print(f"[{r.score:.4f}] {r.record.text}")
# [0.6647] cat
# [0.5364] dog
# [0.4798] tiger

for r in db.search("european nation", top_k=2):
    print(f"[{r.score:.4f}] {r.record.text}")
# [0.6357] france
# [0.5690] germany

for r in db.search("italian food", top_k=2):
    print(f"[{r.score:.4f}] {r.record.text}")
# [0.6339] pasta
# [0.6206] pizza
```

The score is cosine similarity in the range `[-1, 1]`. Higher = more similar. Anything above ~0.5 is a strong match.

### Searching with your trained Word2Vec model

```python
db = VectorDB(embedder=get_embedder("word2vec", model_path="models/word2vec.pt"))

for r in db.search("king", top_k=3):
    print(f"[{r.score:.4f}] {r.record.text}")
# [0.8567] prince
# [0.8407] queen
# [0.7899] princess
```

Note: Word2Vec only knows words it saw during training. Unknown words (OOV) return zero vectors and will score 0.0. If a word you care about is missing, retrain with more tokens (`--max-tokens 5000000`) or a lower `--min-count`.

### Command-line search

Use the `vdb` CLI to add words and search without writing Python — see the [CLI section](#cli--add-search-and-visualize-from-the-terminal) above.

To explore the built-in word cluster demo:

```bash
# Pretrained model
uv run python examples/word_clusters.py

# Your trained Word2Vec
uv run python examples/word_clusters.py --embedder word2vec --model models/word2vec.pt

# Side-by-side comparison of both
uv run python examples/compare_models.py --model models/word2vec.pt
```

---

## Switching embedding models

The `VectorDB` takes any `BaseEmbedder`. Swap at construction time:

```python
from core import VectorDB, get_embedder

# Pretrained sentence-transformers (recommended for quality)
db = VectorDB(embedder=get_embedder("pretrained"))

# Your from-scratch Word2Vec (must train first)
db = VectorDB(embedder=get_embedder("word2vec", model_path="models/word2vec.pt"))

# Or a bigger pretrained model
db = VectorDB(embedder=get_embedder("pretrained", model_name="all-mpnet-base-v2"))
```

---

## Using the vector DB

```python
from core import VectorDB, get_embedder

db = VectorDB(embedder=get_embedder("pretrained"))

# Add documents
db.add("doc1", "The cat sat on the mat")
db.add("doc2", "A dog ran across the field")
db.add("doc3", "Python is a programming language")

# Or add many at once (faster — one forward pass)
db.add_batch([
    {"id": "a", "text": "Paris is the capital of France"},
    {"id": "b", "text": "Berlin is the capital of Germany", "metadata": {"country": "de"}},
])

# Search
results = db.search("feline animals", top_k=3)
for r in results:
    print(f"[{r.score:.3f}] {r.record.text}")

# Save and reload
db.save("my_db/")
db.load("my_db/")
```

---

## Architecture

```
VectorDB
 ├── embedder: BaseEmbedder          ← swappable embedding model
 │    ├── SentenceTransformerEmbedder  (sentence-transformers)
 │    └── Word2VecEmbedder             (our from-scratch model)
 │
 ├── storage: VectorStorage          ← in-memory store
 │    ├── _records: list[VectorRecord] (metadata)
 │    └── _matrix: np.ndarray (n, d)  (all vectors, stacked)
 │
 └── search: brute_force_search      ← cosine similarity scan
      └── _cosine_similarities()       (one matrix multiply)

Training pipeline (independent of VectorDB):
 corpus.py → tokenize → Vocabulary.build() → encode
 dataset.py → SkipGramDataset → (center, context, negatives)
 trainer.py → SkipGramModel → train → save embeddings matrix
```

### Key design decision: why is `_matrix` separate from `_records`?

Cosine similarity search requires multiplying the query vector against every stored vector. If vectors were stored inside VectorRecord objects in a Python list, we'd need to extract them into a temporary array each query. By maintaining a permanently-stacked matrix, the search is always a single `matrix @ query` call — one BLAS operation instead of n Python function calls.

---

## What real vector DBs do differently

This implementation is intentionally simple. Production systems add:

| Feature | This project | Production (Qdrant/Milvus) |
|---|---|---|
| Search | Brute force O(n·d) | HNSW graphs O(log n) |
| Language | Python | Rust / Go |
| Storage | numpy + JSON | Memory-mapped segment files |
| Scale | ~100k vectors | Billions of vectors |
| Filtering | None | Payload filtering (metadata + vector) |
| Quantization | None | Scalar / product quantization (4-8× compression) |

**Further reading:**
- [HNSW paper](https://arxiv.org/abs/1603.09320) — the algorithm inside Qdrant/Weaviate
- [Qdrant architecture](https://qdrant.tech/documentation/concepts/) — well-documented real system
- [FAISS wiki](https://github.com/facebookresearch/faiss/wiki) — Meta's search library
