"""
compare_models.py — Side-by-side comparison of Word2Vec vs pre-trained embedder.

This script runs the same set of words through both models and shows:
    1. Nearest-neighbour comparisons (which words does each model think are similar?)
    2. Side-by-side 2D visualisations

This comparison reveals the limitations of a small Word2Vec model trained on
a limited corpus vs a massive pre-trained transformer model:
    - Word2Vec may cluster words that co-occur frequently regardless of meaning
    - sentence-transformers understands semantic similarity more robustly
    - Both should agree on obvious cases (cat/dog, king/queen)

Usage:
    # Requires a trained Word2Vec model:
    uv run python examples/compare_models.py --model models/word2vec.pt
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from rich import print as rprint
from rich.table import Table
from rich.columns import Columns

from core import VectorDB, get_embedder
from viz.reduce import reduce
from viz.plot import plot_matplotlib


WORDS = [
    # Animals
    "cat", "dog", "lion", "tiger", "wolf", "bird", "fish",
    # Royalty
    "king", "queen", "prince", "princess",
    # Tech
    "computer", "software", "algorithm", "internet",
    # Food
    "pizza", "bread", "rice", "soup",
    # Countries
    "france", "germany", "japan", "china",
]

GROUPS = {
    "cat": "animals", "dog": "animals", "lion": "animals", "tiger": "animals",
    "wolf": "animals", "bird": "animals", "fish": "animals",
    "king": "royalty", "queen": "royalty", "prince": "royalty", "princess": "royalty",
    "computer": "tech", "software": "tech", "algorithm": "tech", "internet": "tech",
    "pizza": "food", "bread": "food", "rice": "food", "soup": "food",
    "france": "countries", "germany": "countries", "japan": "countries", "china": "countries",
}

QUERY_WORDS = ["king", "cat", "computer", "pizza"]


def build_db(embedder) -> VectorDB:
    db = VectorDB(embedder=embedder)
    db.add_batch([
        {"id": w, "text": w, "metadata": {"group": GROUPS[w]}}
        for w in WORDS
    ])
    return db


def print_neighbours(db: VectorDB, model_name: str) -> None:
    rprint(f"\n[bold]{model_name}[/bold]")
    for query in QUERY_WORDS:
        results = db.search(query, top_k=5)
        neighbours = [r.record.text for r in results if r.record.text != query][:4]
        scores     = [r.score for r in results if r.record.text != query][:4]
        table = Table(title=f'"{query}"', show_header=True, header_style="dim")
        table.add_column("Word")
        table.add_column("Cosine sim", style="green")
        for w, s in zip(neighbours, scores):
            table.add_row(w, f"{s:.4f}")
        rprint(table)


def main():
    parser = argparse.ArgumentParser(description="Compare Word2Vec vs pretrained embedder")
    parser.add_argument("--model",       default="models/word2vec.pt", help="Trained Word2Vec .pt file")
    parser.add_argument("--model-name",  default="all-MiniLM-L6-v2")
    parser.add_argument("--method",      default="tsne", choices=["tsne", "pca"])
    args = parser.parse_args()

    # --- Load embedders -----------------------------------------------------
    rprint("[bold cyan]Loading pretrained embedder...[/bold cyan]")
    pretrained_emb = get_embedder("pretrained", model_name=args.model_name)

    rprint("[bold cyan]Loading Word2Vec embedder...[/bold cyan]")
    word2vec_emb = get_embedder("word2vec", model_path=args.model)

    # --- Build both DBs -----------------------------------------------------
    db_pretrained = build_db(pretrained_emb)
    db_word2vec   = build_db(word2vec_emb)

    # --- Print nearest-neighbour comparison ---------------------------------
    rprint("\n" + "="*60)
    rprint("[bold yellow]Nearest Neighbour Comparison[/bold yellow]")
    rprint("="*60)
    print_neighbours(db_pretrained, f"Pretrained ({args.model_name})")
    print_neighbours(db_word2vec,   f"Word2Vec (from scratch)")

    # --- Side-by-side 2D visualisation ---------------------------------------
    rprint(f"\n[bold cyan]Generating side-by-side {args.method.upper()} plots...[/bold cyan]")

    groups = [GROUPS[w] for w in WORDS]

    vectors_pre = db_pretrained.storage.get_matrix()
    vectors_w2v = db_word2vec.storage.get_matrix()

    coords_pre = reduce(vectors_pre, method=args.method)
    coords_w2v = reduce(vectors_w2v, method=args.method)

    # Plot side by side in one figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    unique_groups = sorted(set(groups))
    cmap = plt.cm.get_cmap("tab10", len(unique_groups))
    colour_map = {g: cmap(i) for i, g in enumerate(unique_groups)}

    for ax, coords, title in [
        (ax1, coords_pre, f"Pretrained ({args.model_name})"),
        (ax2, coords_w2v, "Word2Vec (from scratch)"),
    ]:
        for group in unique_groups:
            mask = [i for i, g in enumerate(groups) if g == group]
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=[colour_map[group]], label=group, alpha=0.8, s=80,
            )
        for i, word in enumerate(WORDS):
            ax.annotate(word, (coords[i, 0], coords[i, 1]),
                        fontsize=8, alpha=0.9, xytext=(3, 3),
                        textcoords="offset points")
        ax.set_title(f"{title}\n({args.method.upper()})", fontsize=11, fontweight="bold")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.2)

    plt.suptitle("Word Embeddings: Pretrained vs From Scratch", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
