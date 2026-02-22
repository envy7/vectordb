"""
corpus.py — Text loading, tokenization, and vocabulary building.

This module turns raw text into the integer sequences that a neural network
can actually consume.

The pipeline is:
    raw text → tokens (strings) → vocabulary → token IDs (integers)

Why integers?
    Neural networks operate on numbers. An embedding layer is just a lookup
    table: given index i, return row i of the weight matrix. So "cat" must
    become, say, 1042 before we can look up its embedding.

Vocabulary design:
    We only keep words that appear at least `min_count` times. Rare words:
    - Add noise (not enough examples to learn from)
    - Bloat the embedding matrix (memory cost is vocab_size × embed_dim)
    - Real Word2Vec typically uses min_count=5, subsampling of frequent words

Further reading:
    Original Word2Vec paper: https://arxiv.org/abs/1301.3781
    word2vec tutorial:       https://jalammar.github.io/illustrated-word2vec/
"""

import re
from collections import Counter
from typing import Generator


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    """
    Convert raw text to a list of lowercase word tokens.

    Simple approach: lowercase everything, strip non-alpha characters, split
    on whitespace. Real NLP systems use more sophisticated tokenisers (BPE,
    WordPiece) but this is sufficient for learning Word2Vec concepts.
    """
    text = text.lower()
    # Replace anything that isn't a letter or whitespace with a space
    text = re.sub(r"[^a-z\s]", " ", text)
    # Split on any whitespace, discard empty strings
    return [t for t in text.split() if t]


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

class Vocabulary:
    """
    Bidirectional mapping between words and integer indices.

    After calling build(), you can:
        vocab["cat"]   → 42      (word → index)
        vocab[42]      → "cat"   (index → word)
        len(vocab)     → number of unique words kept
    """

    def __init__(self, min_count: int = 5):
        """
        Args:
            min_count: Drop words that appear fewer than this many times.
                       Lower = larger vocab (more words, more memory, more noise).
                       Higher = smaller vocab (fewer words, faster training).
        """
        self.min_count = min_count
        self.word_to_idx: dict[str, int] = {}
        self.idx_to_word: dict[int, str] = {}
        self.word_counts: Counter = Counter()

    def build(self, tokens: list[str]) -> None:
        """
        Build the vocabulary from a flat list of tokens.

        Words below min_count are discarded. The remaining words are sorted
        alphabetically so the index assignment is deterministic (same corpus
        always produces the same vocab).
        """
        self.word_counts = Counter(tokens)

        # Keep only frequent enough words
        vocab_words = sorted(
            word for word, count in self.word_counts.items()
            if count >= self.min_count
        )

        self.word_to_idx = {word: idx for idx, word in enumerate(vocab_words)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}

        print(f"Vocabulary: {len(tokens):,} tokens → {len(self):,} unique words "
              f"(min_count={self.min_count})")

    def encode(self, tokens: list[str]) -> list[int]:
        """
        Convert a token list to an index list, silently skipping OOV words.

        OOV = Out Of Vocabulary — words below min_count or never seen in training.
        Skipping them (rather than using a special <UNK> token) is the standard
        Word2Vec approach.
        """
        return [self.word_to_idx[t] for t in tokens if t in self.word_to_idx]

    def __len__(self) -> int:
        return len(self.word_to_idx)

    def __contains__(self, word: str) -> bool:
        return word in self.word_to_idx


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def load_corpus(path: str, max_tokens: int | None = None) -> list[str]:
    """
    Load a text file and return a flat list of tokens.

    Args:
        path:       Path to a .txt file
        max_tokens: If set, only return the first N tokens (useful for quick
                    experiments with the full text8 dataset)
    """
    with open(path, encoding="utf-8") as f:
        text = f.read()
    tokens = tokenize(text)
    if max_tokens is not None:
        tokens = tokens[:max_tokens]
    return tokens


# ---------------------------------------------------------------------------
# Sliding window pair generation
# ---------------------------------------------------------------------------

def sliding_window_pairs(
    token_ids: list[int],
    window_size: int = 2,
) -> Generator[tuple[int, int], None, None]:
    """
    Generate (center_word_id, context_word_id) training pairs.

    For Skip-gram Word2Vec, we slide a window over the token sequence.
    For each center word, every word within `window_size` positions is a
    positive training example (they co-occur → should be similar).

    Example with window_size=2 on "the quick brown fox jumps":
        center="quick" → contexts: "the", "brown"   (2 left, right of quick)
        center="brown" → contexts: "the", "quick", "fox", "jumps"

    The number of pairs grows linearly with corpus size and window size.
    Larger window → captures broader topical similarity (less syntactic).
    """
    n = len(token_ids)
    for i, center_id in enumerate(token_ids):
        # Clamp window to array boundaries
        left  = max(0, i - window_size)
        right = min(n, i + window_size + 1)
        for j in range(left, right):
            if j != i:  # skip the center word itself
                yield center_id, token_ids[j]
