"""
dataset.py — PyTorch Dataset for Skip-gram Word2Vec training.

The key innovation in Word2Vec (vs earlier neural LMs) is *negative sampling*.

Without it, training would require:
    softmax over the entire vocabulary for each training pair
    = expensive: O(vocab_size) per step

With negative sampling:
    For each real (center, context) pair, sample k random "negative" words
    that did NOT appear near the center. Train a binary classifier:
        - Is this a real context word? (positive pair → label 1)
        - Is this a random word?      (negative pair → label 0)
    Cost is now O(k) per step, and k is tiny (5-20).

The model learns:
    - Pull real context words CLOSE to the center in embedding space
    - Push random words FAR from the center

This is a form of contrastive learning — the same principle used in
modern models like SimCLR, CLIP, and sentence-transformers!

Further reading:
    Negative sampling explained: https://arxiv.org/abs/1402.3722
    Visual explanation:          https://jalammar.github.io/illustrated-word2vec/
"""

import torch
from torch.utils.data import Dataset

from .corpus import sliding_window_pairs


class SkipGramDataset(Dataset):
    """
    PyTorch Dataset that generates (center, context, negatives) triples.

    Each __getitem__ returns one training example:
        center    : int — index of the center word
        context   : int — index of a true context word (positive example)
        negatives : Tensor[n_negatives] — indices of random words (negative examples)
    """

    def __init__(
        self,
        token_ids: list[int],
        vocab_size: int,
        window_size: int = 2,
        n_negatives: int = 5,
    ):
        """
        Args:
            token_ids:   Encoded corpus (list of integer word indices)
            vocab_size:  Total vocabulary size (for uniform negative sampling)
            window_size: Context window radius (2 → look 2 words left and right)
            n_negatives: Number of random negatives per positive pair
                         More negatives → slower but more stable training
        """
        print("Generating training pairs (this may take a moment for large corpora)...")
        # Pre-compute all pairs upfront and store as a list of tuples
        # For very large corpora you'd generate pairs on-the-fly in __getitem__
        self.pairs = list(sliding_window_pairs(token_ids, window_size))
        self.vocab_size = vocab_size
        self.n_negatives = n_negatives
        print(f"Generated {len(self.pairs):,} training pairs")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        center, context = self.pairs[idx]

        # Sample n_negatives random word indices uniformly from the vocabulary.
        # In the original Word2Vec, negatives are sampled proportional to
        # word frequency^(3/4) — that's a small refinement we skip here.
        negatives = torch.randint(0, self.vocab_size, (self.n_negatives,))

        return {
            "center":    torch.tensor(center,  dtype=torch.long),
            "context":   torch.tensor(context, dtype=torch.long),
            "negatives": negatives,
        }
