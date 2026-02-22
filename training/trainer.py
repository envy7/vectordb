"""
trainer.py — Train a Word2Vec Skip-gram model from scratch.

Model architecture:
    Two embedding matrices (both learned during training):
        input_embeddings  : shape (vocab_size, embed_dim) — the "center word" side
        output_embeddings : shape (vocab_size, embed_dim) — the "context word" side

    Having two separate matrices is a design choice from the original Word2Vec
    paper. It prevents a word from being its own best context and improves
    gradient flow. After training, only input_embeddings is used for inference.

Loss function: Negative Sampling Loss
    For a real (center, context) pair:
        maximize: sigmoid(center · context)   ← pull them together
        minimize: sigmoid(center · negative)  ← push randoms apart

    Equivalently: minimize -[log σ(pos_score) + Σ log σ(-neg_score)]

    This is binary cross-entropy: "is this pair real (1) or fake (0)?"

Training tips:
    - Larger embed_dim → richer representations but more memory + slower
    - More epochs → better quality but diminishing returns after ~5
    - window_size=2 captures syntactic patterns; window_size=5 captures semantics
    - min_count filters noise; 5 is a good default

Usage (CLI):
    uv run train-word2vec --corpus data/corpus.txt --output models/word2vec.pt

Usage (Python):
    from training.trainer import train
    train(corpus_path="data/corpus.txt", output_path="models/word2vec.pt")

Further reading:
    Original paper: https://arxiv.org/abs/1301.3781
    word2vec parameter tuning: https://rare-technologies.com/word2vec-tutorial/
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .corpus import Vocabulary, load_corpus
from .dataset import SkipGramDataset


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class SkipGramModel(nn.Module):
    """
    The Skip-gram Word2Vec neural network.

    This is a remarkably simple model — just two embedding (lookup) tables.
    The "magic" is entirely in what the loss function forces the model to learn.

    Diagram:
        center_word_id → input_embeddings[center_id]    = center_vec  (dim,)
        context_word_id → output_embeddings[context_id] = context_vec (dim,)
        negative_ids → output_embeddings[neg_ids]       = neg_vecs    (k, dim)

        loss = -log σ(center_vec · context_vec)
               - Σ log σ(-center_vec · neg_vec)
    """

    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        # Two separate embedding matrices as described above
        self.input_embeddings  = nn.Embedding(vocab_size, embed_dim)
        self.output_embeddings = nn.Embedding(vocab_size, embed_dim)

        # Small random init for input; zeros for output (common practice)
        nn.init.uniform_(self.input_embeddings.weight,  -0.5 / embed_dim, 0.5 / embed_dim)
        nn.init.zeros_(self.output_embeddings.weight)

    def forward(
        self,
        center:    torch.Tensor,  # (batch,)
        context:   torch.Tensor,  # (batch,)
        negatives: torch.Tensor,  # (batch, n_negatives)
    ) -> torch.Tensor:
        """
        Compute negative sampling loss for a batch.

        Returns a scalar tensor — the mean loss over the batch.
        """
        # Look up embeddings for each word in the batch
        center_emb  = self.input_embeddings(center)    # (batch, dim)
        context_emb = self.output_embeddings(context)  # (batch, dim)
        neg_emb     = self.output_embeddings(negatives) # (batch, n_neg, dim)

        # Positive score: dot product of center and real context
        # sum(dim=1) does element-wise multiply then sums → dot product per item
        pos_score = torch.sum(center_emb * context_emb, dim=1)  # (batch,)
        pos_loss  = torch.nn.functional.logsigmoid(pos_score)   # (batch,)

        # Negative score: dot product of center with each negative
        # unsqueeze adds a dim: (batch, dim) → (batch, dim, 1)
        # bmm does batch matrix multiply: (batch, n_neg, dim) @ (batch, dim, 1) = (batch, n_neg, 1)
        neg_score = torch.bmm(neg_emb, center_emb.unsqueeze(2)).squeeze(2)  # (batch, n_neg)
        # We want neg scores to be LOW → use -neg_score in logsigmoid
        neg_loss  = torch.nn.functional.logsigmoid(-neg_score).sum(dim=1)  # (batch,)

        # Mean over batch (negative sign because we minimise loss)
        return -(pos_loss + neg_loss).mean()


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

def train(
    corpus_path:   str,
    output_path:   str,
    embed_dim:     int   = 100,
    window_size:   int   = 2,
    n_negatives:   int   = 5,
    batch_size:    int   = 1024,
    epochs:        int   = 5,
    min_count:     int   = 5,
    max_tokens:    int | None = 1_000_000,
    learning_rate: float = 0.001,
) -> None:
    """
    Train a Word2Vec model and save the result to `output_path`.

    The saved file is a torch checkpoint dict containing:
        embeddings  : np.ndarray (vocab_size, embed_dim)
        word_to_idx : dict
        idx_to_word : dict
        embed_dim   : int
    """
    t0 = time.time()

    # --- Load and tokenise corpus -------------------------------------------
    print(f"\n{'='*60}")
    print(f"  Word2Vec Training")
    print(f"{'='*60}")
    print(f"\n[1/5] Loading corpus from: {corpus_path}")
    tokens = load_corpus(corpus_path, max_tokens=max_tokens)
    print(f"      {len(tokens):,} tokens loaded")

    # --- Build vocabulary ----------------------------------------------------
    print(f"\n[2/5] Building vocabulary (min_count={min_count})")
    vocab = Vocabulary(min_count=min_count)
    vocab.build(tokens)

    # --- Encode corpus to integers ------------------------------------------
    print(f"\n[3/5] Encoding corpus")
    token_ids = vocab.encode(tokens)
    print(f"      {len(token_ids):,} token IDs (after OOV removal)")

    # --- Create dataset and dataloader --------------------------------------
    print(f"\n[4/5] Building dataset")
    dataset = SkipGramDataset(
        token_ids=token_ids,
        vocab_size=len(vocab),
        window_size=window_size,
        n_negatives=n_negatives,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,   # set >0 for multi-process loading on large datasets
        pin_memory=True,
    )

    # --- Setup model and optimizer ------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "mps"
                          if torch.backends.mps.is_available() else "cpu")
    print(f"\n[5/5] Training on: {device}")
    print(f"      vocab_size={len(vocab):,} | embed_dim={embed_dim} | "
          f"epochs={epochs} | batch_size={batch_size}")

    model = SkipGramModel(len(vocab), embed_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # --- Training loop -------------------------------------------------------
    print()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        epoch_start = time.time()

        for batch_idx, batch in enumerate(loader):
            center    = batch["center"].to(device)
            context   = batch["context"].to(device)
            negatives = batch["negatives"].to(device)

            optimizer.zero_grad()
            loss = model(center, context, negatives)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Print progress every 200 batches
            if batch_idx % 200 == 0:
                pct = 100 * batch_idx / len(loader)
                print(f"  Epoch {epoch}/{epochs} | {pct:5.1f}% | loss={loss.item():.4f}", end="\r")

        avg_loss = epoch_loss / len(loader)
        elapsed  = time.time() - epoch_start
        print(f"  Epoch {epoch}/{epochs} done | avg_loss={avg_loss:.4f} | {elapsed:.1f}s")

    # --- Extract and save embeddings ----------------------------------------
    print(f"\nExtracting embedding matrix...")
    # Detach from computation graph, move to CPU, convert to numpy
    embeddings = model.input_embeddings.weight.detach().cpu().numpy()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save({
        "embeddings":  embeddings,            # numpy array
        "word_to_idx": vocab.word_to_idx,
        "idx_to_word": vocab.idx_to_word,
        "embed_dim":   embed_dim,
    }, output_path)

    total_time = time.time() - t0
    print(f"\nSaved to: {output_path}")
    print(f"Total training time: {total_time:.1f}s")
    print(f"Embedding matrix: {embeddings.shape}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train Word2Vec from scratch on a text corpus"
    )
    parser.add_argument("--corpus",     default="data/corpus.txt",      help="Path to training corpus")
    parser.add_argument("--output",     default="models/word2vec.pt",   help="Where to save the model")
    parser.add_argument("--embed-dim",  type=int, default=100,          help="Embedding dimensions")
    parser.add_argument("--window",     type=int, default=2,            help="Context window size")
    parser.add_argument("--negatives",  type=int, default=5,            help="Negative samples per pair")
    parser.add_argument("--batch-size", type=int, default=1024,         help="Training batch size")
    parser.add_argument("--epochs",     type=int, default=5,            help="Training epochs")
    parser.add_argument("--min-count",  type=int, default=5,            help="Min word frequency")
    parser.add_argument("--max-tokens", type=int, default=1_000_000,    help="Max tokens to load (0=all)")
    parser.add_argument("--lr",         type=float, default=0.001,      help="Learning rate")
    args = parser.parse_args()

    train(
        corpus_path=args.corpus,
        output_path=args.output,
        embed_dim=args.embed_dim,
        window_size=args.window,
        n_negatives=args.negatives,
        batch_size=args.batch_size,
        epochs=args.epochs,
        min_count=args.min_count,
        max_tokens=args.max_tokens or None,
        learning_rate=args.lr,
    )


if __name__ == "__main__":
    main()
