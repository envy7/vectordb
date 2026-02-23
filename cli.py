"""
cli.py — Command-line interface for the VectorDB project.

Commands:
    vdb add <word> [word2 ...]     Add words or phrases to the database
    vdb search <query>             Find nearest neighbors
    vdb list                       Show all stored words
    vdb viz                        Visualize all words in 2D
    vdb clear                      Remove all words from the database

The database is persisted to disk (default: .vdb/) so it survives
between command invocations. The embedder is chosen once at DB creation
time and stored in .vdb/config.json.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rich import print as rprint
from rich.table import Table

from core import VectorDB, get_embedder
from core.storage import VectorStorage

DEFAULT_DB = ".vdb"


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _config_path(db_dir: str) -> str:
    return os.path.join(db_dir, "config.json")


def _load_config(db_dir: str) -> dict | None:
    path = _config_path(db_dir)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def _save_config(db_dir: str, config: dict) -> None:
    os.makedirs(db_dir, exist_ok=True)
    with open(_config_path(db_dir), "w") as f:
        json.dump(config, f, indent=2)


def _make_embedder(config: dict):
    name = config.get("embedder", "pretrained")
    if name == "pretrained":
        return get_embedder("pretrained", model_name=config.get("model_name", "all-MiniLM-L6-v2"))
    elif name == "word2vec":
        return get_embedder("word2vec", model_path=config.get("model_path", "models/word2vec.pt"))
    else:
        rprint(f"[red]Unknown embedder: {name}[/red]")
        sys.exit(1)


# ---------------------------------------------------------------------------
# DB load/init
# ---------------------------------------------------------------------------

def _open_db(db_dir: str, args) -> tuple[VectorDB, dict]:
    """
    Load an existing DB from disk, or initialise a new one from args.
    Returns (db, config).
    """
    config = _load_config(db_dir)

    if config is None:
        # First use — create config from CLI flags
        config = {
            "embedder": args.embedder,
            "model_name": args.model_name,
            "model_path": args.model,
        }

    embedder = _make_embedder(config)
    db = VectorDB(embedder=embedder)

    vectors_path = os.path.join(db_dir, "vectors.npy")
    metadata_path = os.path.join(db_dir, "metadata.json")
    if os.path.exists(vectors_path) and os.path.exists(metadata_path):
        # Suppress the built-in print from VectorStorage.load
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            db.load(db_dir)

    return db, config


def _require_db(db_dir: str) -> None:
    """Exit with an error if the DB directory does not exist."""
    if not os.path.exists(db_dir):
        rprint(f"[red]No database found at '{db_dir}'. Run 'vdb add <words>' first.[/red]")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_add(args):
    db, config = _open_db(args.db, args)

    existing = {r.id for r in db.storage.get_records()}
    group = args.group or "custom"

    to_add = []
    skipped = []
    for word in args.words:
        if word in existing:
            skipped.append(word)
        else:
            to_add.append({"id": word, "text": word, "metadata": {"group": group}})

    if skipped:
        rprint(f"[yellow]Already in DB (skipped):[/yellow] {', '.join(skipped)}")

    if not to_add:
        rprint("[dim]Nothing new to add.[/dim]")
        return

    rprint(f"[cyan]Embedding {len(to_add)} word(s)...[/cyan]")
    db.add_batch(to_add)

    _save_config(args.db, config)
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        db.save(args.db)

    rprint(f"[green]Added {len(to_add)} word(s).[/green] DB now has [bold]{len(db)}[/bold] entries.")


def cmd_search(args):
    _require_db(args.db)
    db, _ = _open_db(args.db, args)

    if len(db) == 0:
        rprint("[yellow]Database is empty. Use 'vdb add' to add words.[/yellow]")
        return

    algo = "HNSW" if db.using_hnsw else "brute force"
    top_k = min(args.top_k, len(db))
    results = db.search(args.query, top_k=top_k, ef=args.ef)

    table = Table(
        title=f'Nearest to "[bold]{args.query}[/bold]"  [dim]({algo})[/dim]',
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Rank", style="dim", width=6)
    table.add_column("Word")
    table.add_column("Score", style="green")
    table.add_column("Group", style="dim")

    for r in results:
        group = r.record.metadata.get("group", "")
        table.add_row(str(r.rank), r.record.text, f"{r.score:.4f}", group)

    rprint(table)


def cmd_index(args):
    _require_db(args.db)
    db, config = _open_db(args.db, args)

    if len(db) == 0:
        rprint("[yellow]Database is empty. Add words first with 'vdb add'.[/yellow]")
        return

    db.build_index(M=args.M, ef_construction=args.ef_construction)

    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        db.save(args.db)

    rprint(
        f"[green]HNSW index saved.[/green] "
        f"Future searches will use approximate search instead of brute force."
    )


def cmd_list(args):
    _require_db(args.db)
    db, config = _open_db(args.db, args)
    records = db.storage.get_records()

    if not records:
        rprint("[yellow]Database is empty.[/yellow]")
        return

    # Group entries by their metadata group
    groups: dict[str, list[str]] = {}
    for r in records:
        g = r.metadata.get("group", "ungrouped")
        groups.setdefault(g, []).append(r.text)

    rprint(
        f"\n[bold]Database:[/bold] {args.db}  |  "
        f"[bold]{len(records)}[/bold] entries  |  "
        f"embedder: [cyan]{config['embedder']}[/cyan]\n"
    )
    for group, words in groups.items():
        rprint(f"  [cyan]{group}[/cyan]: {', '.join(words)}")
    rprint()


def cmd_viz(args):
    _require_db(args.db)
    db, config = _open_db(args.db, args)
    records = db.storage.get_records()

    if len(records) < 3:
        rprint(f"[yellow]Need at least 3 words to visualize (have {len(records)}).[/yellow]")
        return

    from viz.reduce import reduce
    from viz.plot import plot_matplotlib, plot_plotly, plot_plotly_3d

    words = [r.text for r in records]
    groups = [r.metadata.get("group", "custom") for r in records]
    vectors = db.storage.get_matrix()
    dims = args.dims

    method = args.method
    if method == "tsne" and len(records) < 10:
        rprint(f"[yellow]Too few words for t-SNE (need ≥10), switching to PCA.[/yellow]")
        method = "pca"

    rprint(f"[cyan]Reducing {len(records)} vectors to {dims}D with {method.upper()}...[/cyan]")
    coords = reduce(vectors, method=method, n_components=dims)

    title = f"Word Map — {config['embedder']} — {method.upper()} {dims}D"
    if dims == 3:
        plot_plotly_3d(coords, labels=words, groups=groups, title=title)
    elif args.interactive:
        plot_plotly(coords, labels=words, groups=groups, title=title)
    else:
        plot_matplotlib(
            coords, labels=words, groups=groups, title=title, save_path=args.save
        )


def cmd_clear(args):
    import shutil

    if not os.path.exists(args.db):
        rprint(f"[yellow]No database at '{args.db}', nothing to clear.[/yellow]")
        return

    # Count entries from metadata.json without loading the embedder
    count = 0
    meta_path = os.path.join(args.db, "metadata.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                count = len(json.load(f))
        except Exception:
            pass

    shutil.rmtree(args.db)
    rprint(f"[green]Cleared database ({count} entries removed).[/green]")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="vdb",
        description="VectorDB — add, search, and visualize words in embedding space.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  vdb add cat dog wolf lion tiger
  vdb add "machine learning" "neural network" --group ai
  vdb search "big cat"
  vdb search wolf --top-k 10
  vdb list
  vdb viz --interactive
  vdb viz --method pca --save plot.png
  vdb clear
""",
    )

    # Global flags (apply to all subcommands)
    parser.add_argument(
        "--db", default=DEFAULT_DB,
        help=f"Database directory (default: {DEFAULT_DB})",
    )
    parser.add_argument(
        "--embedder", default="pretrained", choices=["pretrained", "word2vec"],
        help="Embedder — only used when creating a new DB (default: pretrained)",
    )
    parser.add_argument(
        "--model", default="models/word2vec.pt",
        help="Path to Word2Vec checkpoint (used with --embedder word2vec)",
    )
    parser.add_argument(
        "--model-name", default="all-MiniLM-L6-v2", dest="model_name",
        help="sentence-transformers model name (used with --embedder pretrained)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # add
    p_add = sub.add_parser("add", help="Add words or phrases to the database")
    p_add.add_argument("words", nargs="+", help="Words or phrases to embed and store")
    p_add.add_argument("--group", default=None, help="Tag these words with a group label")

    # search
    p_search = sub.add_parser("search", help="Find nearest neighbors for a query")
    p_search.add_argument("query", help="Search term (word or phrase)")
    p_search.add_argument(
        "--top-k", type=int, default=5, dest="top_k",
        help="Number of results to show (default: 5)",
    )
    p_search.add_argument(
        "--ef", type=int, default=50,
        help="HNSW search candidate list size — higher = better recall, slower (default: 50)",
    )

    # list
    sub.add_parser("list", help="List all words currently in the database")

    # viz
    p_viz = sub.add_parser("viz", help="Visualize words in 2D embedding space")
    p_viz.add_argument(
        "--interactive", action="store_true",
        help="Open an interactive Plotly chart in the browser",
    )
    p_viz.add_argument(
        "--method", default="tsne", choices=["tsne", "pca"],
        help="Dimensionality reduction method (default: tsne)",
    )
    p_viz.add_argument(
        "--dims", type=int, default=2, choices=[2, 3],
        help="2 or 3 dimensions (default: 2). 3D always opens in the browser.",
    )
    p_viz.add_argument("--save", default=None, help="Save the plot to a PNG file (2D only)")

    # index
    p_index = sub.add_parser(
        "index", help="Build an HNSW index for fast approximate search"
    )
    p_index.add_argument(
        "--M", type=int, default=16,
        help="Connections per node per layer — higher = better recall, more memory (default: 16)",
    )
    p_index.add_argument(
        "--ef-construction", type=int, default=200, dest="ef_construction",
        help="Candidate list size during build — higher = better graph, slower build (default: 200)",
    )

    # clear
    sub.add_parser("clear", help="Delete all words from the database")

    args = parser.parse_args()

    {
        "add":    cmd_add,
        "search": cmd_search,
        "list":   cmd_list,
        "viz":    cmd_viz,
        "index":  cmd_index,
        "clear":  cmd_clear,
    }[args.command](args)


if __name__ == "__main__":
    main()
