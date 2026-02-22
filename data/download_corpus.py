"""
download_corpus.py — Download the text8 dataset for Word2Vec training.

text8 is the standard benchmark corpus for word embedding models.
It's the first 100 million characters of a cleaned Wikipedia dump (2006).
Created by Matt Mahoney: http://mattmahoney.net/dc/textdata.html

The file is ~31 MB compressed, ~100 MB uncompressed.
One million tokens is enough to train a decent Word2Vec model in a few minutes.

Usage:
    uv run python data/download_corpus.py
    uv run python data/download_corpus.py --max-tokens 2000000
"""

import argparse
import io
import os
import zipfile

import requests

TEXT8_URL = "http://mattmahoney.net/dc/text8.zip"
OUTPUT_PATH = "data/corpus.txt"


def download_text8(max_tokens: int | None = 1_000_000) -> None:
    os.makedirs("data", exist_ok=True)

    if os.path.exists(OUTPUT_PATH):
        print(f"Corpus already exists at {OUTPUT_PATH}. Delete it to re-download.")
        return

    print(f"Downloading text8 from {TEXT8_URL} ...")
    response = requests.get(TEXT8_URL, stream=True)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    downloaded = 0
    chunks = []

    for chunk in response.iter_content(chunk_size=8192):
        chunks.append(chunk)
        downloaded += len(chunk)
        if total:
            pct = 100 * downloaded / total
            print(f"  {pct:.1f}% ({downloaded // 1024 // 1024} MB)", end="\r")

    print(f"\nDownload complete. Extracting...")

    # text8.zip contains a single file named "text8"
    zip_bytes = b"".join(chunks)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        raw = zf.read("text8").decode("utf-8")

    if max_tokens is not None:
        # text8 is space-separated — split to get tokens, take first N, rejoin
        tokens = raw.split()
        print(f"Full corpus: {len(tokens):,} tokens")
        tokens = tokens[:max_tokens]
        raw = " ".join(tokens)
        print(f"Using first {max_tokens:,} tokens")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(raw)

    size_mb = os.path.getsize(OUTPUT_PATH) / 1024 / 1024
    print(f"Saved to {OUTPUT_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-tokens", type=int, default=1_000_000,
        help="How many tokens to keep (0 = full 17M token dataset)"
    )
    args = parser.parse_args()
    download_text8(max_tokens=args.max_tokens or None)
