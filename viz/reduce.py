"""
reduce.py — Dimensionality reduction for visualising embeddings.

Embeddings live in high-dimensional space (100–1536 dims). We can't directly
plot that, so we project down to 2D using dimensionality reduction.

Two algorithms are available:

PCA (Principal Component Analysis):
    Linear method. Finds the directions (principal components) that capture
    the most variance in the data. Fast and deterministic.

    Limitation: embeddings don't vary linearly — "king - man + woman ≈ queen"
    is a non-linear relationship. PCA flattens these curves and loses structure.
    Good for a quick sanity check.

t-SNE (t-distributed Stochastic Neighbor Embedding):
    Non-linear method. Tries to preserve LOCAL neighbourhoods: words that
    are close in high-dim space should be close in 2D too.

    It does this by:
    1. Computing a probability distribution over pairs in high-dim space
       (close pairs → high probability)
    2. Doing the same in 2D, but using a t-distribution (heavier tails)
    3. Minimising the KL divergence between the two distributions

    The t-distribution prevents the "crowding problem" — without it, distant
    points collapse into the centre in 2D.

    Excellent for revealing semantic clusters. Slow for large datasets.
    Results depend on `perplexity` and random seed — not fully deterministic.

Further reading:
    t-SNE paper:       https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf
    t-SNE explained:   https://distill.pub/2016/misread-tsne/  ← highly recommended visual
    PCA tutorial:      https://builtin.com/data-science/step-step-explanation-principal-component-analysis
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def reduce_pca(vectors: np.ndarray, n_components: int = 2) -> np.ndarray:
    """
    Project vectors to `n_components` dimensions using PCA.

    Args:
        vectors:      Shape (n, dim) — the high-dimensional embeddings
        n_components: Target dimensions (2 for 2D plot, 3 for 3D)

    Returns:
        Shape (n, n_components) — the projected coordinates
    """
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(vectors)


def reduce_tsne(
    vectors:      np.ndarray,
    n_components: int   = 2,
    perplexity:   float = 30.0,
    max_iter:     int   = 1000,
) -> np.ndarray:
    """
    Project vectors to `n_components` dimensions using t-SNE.

    Args:
        vectors:      Shape (n, dim)
        n_components: Target dimensions (usually 2)
        perplexity:   Roughly "effective number of neighbours to consider".
                      Rule of thumb: sqrt(n) is a reasonable starting point.
                      Range: 5–50. Smaller datasets → lower perplexity.
        max_iter:     Optimisation iterations. More → better but slower.

    Returns:
        Shape (n, n_components)

    Note: t-SNE is slow for n > 5000. For large datasets, run PCA first
    to reduce to ~50 dims, then t-SNE on the PCA output. (Sklearn does
    this automatically if n > 30.)
    """
    # If we have many dimensions, PCA first (faster + t-SNE works better).
    # Cap n_components so it never exceeds n_samples or n_features.
    input_vectors = vectors
    if vectors.shape[1] > 50:
        pca_components = min(50, vectors.shape[0] - 1, vectors.shape[1])
        input_vectors = reduce_pca(vectors, n_components=pca_components)

    tsne = TSNE(
        n_components=n_components,
        perplexity=min(perplexity, len(vectors) - 1),  # perplexity < n
        max_iter=max_iter,
        random_state=42,  # fix seed for reproducibility
        init="pca",       # PCA init is more stable than random
    )
    return tsne.fit_transform(input_vectors)


def reduce(
    vectors:  np.ndarray,
    method:   str   = "tsne",
    **kwargs,
) -> np.ndarray:
    """
    Convenience wrapper — choose method by name.

    Args:
        vectors: Shape (n, dim)
        method:  "pca" or "tsne"
        **kwargs: Passed through to the underlying function
    """
    if method == "pca":
        return reduce_pca(vectors, **kwargs)
    elif method == "tsne":
        return reduce_tsne(vectors, **kwargs)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'pca' or 'tsne'.")
