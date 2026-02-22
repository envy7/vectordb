"""
word_clusters.py — Visualise semantic word clusters in embedding space.

This example demonstrates the core promise of embeddings:
    words with similar meanings should be geometrically close.

We add words from five semantic categories to the vector DB, then
visualise them with t-SNE. If the model has learned meaningful embeddings,
each colour cluster should be spatially coherent.

Usage:
    # Using the pretrained model (recommended, no training needed):
    uv run python examples/word_clusters.py

    # Using your trained Word2Vec model:
    uv run python examples/word_clusters.py --embedder word2vec --model models/word2vec.pt

    # Use PCA instead of t-SNE (faster, less informative):
    uv run python examples/word_clusters.py --method pca

    # Interactive plot in the browser:
    uv run python examples/word_clusters.py --interactive
"""

import argparse
import sys
import os

# Make sure we can import from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from rich import print as rprint
from rich.table import Table

from core import VectorDB, get_embedder
from viz.reduce import reduce
from viz.plot import plot_matplotlib, plot_plotly


# ---------------------------------------------------------------------------
# Word groups — each group is a semantic category
# ---------------------------------------------------------------------------

WORD_GROUPS = {
    "animals": [
        "cat", "dog", "fish", "bird", "horse",
        "elephant", "lion", "tiger", "wolf", "rabbit",
    ],
    "countries": [
        "france", "germany", "italy", "spain", "japan",
        "china", "brazil", "india", "canada", "australia",
    ],
    "royalty": [
        "king", "queen", "prince", "princess",
        "duke", "earl", "emperor", "throne", "crown", "palace",
    ],
    "technology": [
        "computer", "software", "internet", "algorithm",
        "database", "network", "processor", "memory", "keyboard", "mouse",
    ],
    "food": [
        "pizza", "pasta", "bread", "rice", "sushi",
        "burger", "salad", "soup", "cheese", "chocolate",
    ],
}


def main():
    parser = argparse.ArgumentParser(description="Visualise word clusters in embedding space")
    parser.add_argument("--embedder",    default="pretrained", choices=["pretrained", "word2vec"])
    parser.add_argument("--model",       default="models/word2vec.pt", help="Word2Vec model path")
    parser.add_argument("--model-name",  default="all-MiniLM-L6-v2",  help="sentence-transformers model")
    parser.add_argument("--method",      default="tsne",               choices=["tsne", "pca"])
    parser.add_argument("--interactive", action="store_true",          help="Open interactive plotly chart")
    parser.add_argument("--save",        default=None,                 help="Save plot to this path (PNG)")
    args = parser.parse_args()

    # --- Load embedder -------------------------------------------------------
    rprint(f"\n[bold cyan]Loading embedder:[/bold cyan] {args.embedder}")
    if args.embedder == "pretrained":
        embedder = get_embedder("pretrained", model_name=args.model_name)
    else:
        embedder = get_embedder("word2vec", model_path=args.model)

    # --- Populate the vector DB ---------------------------------------------
    rprint(f"\n[bold cyan]Embedding words...[/bold cyan]")
    db = VectorDB(embedder=embedder)

    all_words  = []
    all_groups = []
    items = []

    for group, words in WORD_GROUPS.items():
        for word in words:
            items.append({"id": word, "text": word, "metadata": {"group": group}})
            all_words.append(word)
            all_groups.append(group)

    db.add_batch(items)
    rprint(f"  {len(db)} words added | dim={embedder.dim}")

    # --- Show nearest neighbours for a few query words ----------------------
    rprint(f"\n[bold cyan]Nearest neighbours:[/bold cyan]")
    for query in ["king", "cat", "pizza", "computer"]:
        results = db.search(query, top_k=4)
        table = Table(title=f'Similar to "{query}"', show_header=True)
        table.add_column("Rank", style="dim")
        table.add_column("Word")
        table.add_column("Score", style="green")
        for r in results:
            if r.record.text != query:  # skip exact match
                table.add_row(str(r.rank), r.record.text, f"{r.score:.4f}")
        rprint(table)

    # --- Reduce to 2D and plot ----------------------------------------------
    rprint(f"\n[bold cyan]Reducing to 2D with {args.method.upper()}...[/bold cyan]")

    # Extract the vector matrix from storage (same order as all_words)
    vectors = db.storage.get_matrix()  # shape: (n, dim)

    coords_2d = reduce(vectors, method=args.method)

    title = (f"Word Clusters — {embedder.name} — {args.method.upper()}")

    rprint(f"[bold cyan]Plotting...[/bold cyan]")
    if args.interactive:
        plot_plotly(coords_2d, labels=all_words, groups=all_groups, title=title)
    else:
        plot_matplotlib(
            coords_2d,
            labels=all_words,
            groups=all_groups,
            title=title,
            save_path=args.save,
        )


if __name__ == "__main__":
    main()
