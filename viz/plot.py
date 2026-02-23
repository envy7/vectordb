"""
plot.py — Visualise 2D and 3D embedding projections.

Three flavours:
    plot_matplotlib:  static 2D, good for saving to PNG / embedding in notebooks
    plot_plotly:      interactive 2D — hover to see labels, zoom, pan in the browser
    plot_plotly_3d:   interactive 3D — rotate, zoom, hover in the browser

2D functions expect coords of shape (n, 2).
3D function expects coords of shape (n, 3).
All accept the same labels / groups arguments.

Colour-coding by group is the key insight for understanding embeddings:
    if words from the same semantic category (animals, countries, etc.)
    cluster together with the same colour, the model has learned meaningful
    structure.
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# Matplotlib (static)
# ---------------------------------------------------------------------------

def plot_matplotlib(
    coords:   np.ndarray,
    labels:   list[str],
    title:    str             = "Embedding Space",
    groups:   Optional[list[str]] = None,
    figsize:  tuple[int, int] = (14, 10),
    fontsize: int             = 8,
    save_path: Optional[str]  = None,
) -> None:
    """
    Static scatter plot with word labels.

    Args:
        coords:     (n, 2) array of 2D coordinates
        labels:     Text label for each point
        title:      Plot title
        groups:     Category string for each point (used for colour coding)
        figsize:    Figure size in inches
        fontsize:   Label font size
        save_path:  If set, save the figure to this path instead of showing
    """
    fig, ax = plt.subplots(figsize=figsize)

    if groups is not None:
        # Assign a colour to each unique group
        unique_groups = sorted(set(groups))
        # tab10 has 10 distinct colours; tab20 has 20
        cmap = plt.cm.get_cmap("tab10", len(unique_groups))
        colour_map = {g: cmap(i) for i, g in enumerate(unique_groups)}

        for group in unique_groups:
            mask = [i for i, g in enumerate(groups) if g == group]
            xs = coords[mask, 0]
            ys = coords[mask, 1]
            ax.scatter(xs, ys, c=[colour_map[group]], label=group, alpha=0.8, s=60)
        ax.legend(loc="best", fontsize=9)
    else:
        ax.scatter(coords[:, 0], coords[:, 1], alpha=0.8, s=60)

    # Add text label for every point
    for i, label in enumerate(labels):
        ax.annotate(
            label,
            (coords[i, 0], coords[i, 1]),
            fontsize=fontsize,
            alpha=0.9,
            xytext=(3, 3),
            textcoords="offset points",
        )

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Plotly (interactive)
# ---------------------------------------------------------------------------

def plot_plotly(
    coords:  np.ndarray,
    labels:  list[str],
    title:   str                  = "Embedding Space",
    groups:  Optional[list[str]] = None,
    width:   int                  = 1100,
    height:  int                  = 750,
) -> None:
    """
    Interactive scatter plot in the browser (or inline in Jupyter).

    Features:
        - Hover over a point to see its label and group
        - Click legend items to hide/show groups
        - Box or lasso select to isolate a cluster
        - Zoom and pan

    Args:
        coords:  (n, 2) array of 2D coordinates
        labels:  Text label for each point
        title:   Plot title
        groups:  Category string for each point (colour coding)
    """
    fig = go.Figure()

    if groups is not None:
        unique_groups = sorted(set(groups))
        for group in unique_groups:
            mask = [i for i, g in enumerate(groups) if g == group]
            fig.add_trace(go.Scatter(
                x=coords[mask, 0],
                y=coords[mask, 1],
                mode="markers+text",
                name=group,
                text=[labels[i] for i in mask],
                textposition="top center",
                marker=dict(size=9),
                hovertemplate="<b>%{text}</b><br>group: " + group + "<extra></extra>",
            ))
    else:
        fig.add_trace(go.Scatter(
            x=coords[:, 0],
            y=coords[:, 1],
            mode="markers+text",
            text=labels,
            textposition="top center",
            marker=dict(size=9),
            hovertemplate="<b>%{text}</b><extra></extra>",
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        width=width,
        height=height,
        hovermode="closest",
        xaxis=dict(title="Component 1", showgrid=True, gridwidth=0.5, gridcolor="#eee"),
        yaxis=dict(title="Component 2", showgrid=True, gridwidth=0.5, gridcolor="#eee"),
        plot_bgcolor="white",
    )
    fig.show()


# ---------------------------------------------------------------------------
# Plotly 3D (interactive, rotatable)
# ---------------------------------------------------------------------------

def plot_plotly_3d(
    coords:  np.ndarray,
    labels:  list[str],
    title:   str                  = "Embedding Space",
    groups:  Optional[list[str]] = None,
    width:   int                  = 1100,
    height:  int                  = 800,
) -> None:
    """
    Rotatable 3D scatter plot in the browser.

    Features:
        - Click and drag to rotate the point cloud
        - Scroll to zoom
        - Hover over a point to see its label and group
        - Click legend items to hide/show groups

    Args:
        coords:  (n, 3) array of 3D coordinates
        labels:  Text label for each point
        title:   Plot title
        groups:  Category string for each point (colour coding)
    """
    fig = go.Figure()

    if groups is not None:
        unique_groups = sorted(set(groups))
        for group in unique_groups:
            mask = [i for i, g in enumerate(groups) if g == group]
            fig.add_trace(go.Scatter3d(
                x=coords[mask, 0],
                y=coords[mask, 1],
                z=coords[mask, 2],
                mode="markers+text",
                name=group,
                text=[labels[i] for i in mask],
                textposition="top center",
                marker=dict(size=5),
                hovertemplate="<b>%{text}</b><br>group: " + group + "<extra></extra>",
            ))
    else:
        fig.add_trace(go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode="markers+text",
            text=labels,
            textposition="top center",
            marker=dict(size=5),
            hovertemplate="<b>%{text}</b><extra></extra>",
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        width=width,
        height=height,
        scene=dict(
            xaxis_title="Component 1",
            yaxis_title="Component 2",
            zaxis_title="Component 3",
            xaxis=dict(showgrid=True, gridcolor="#ddd"),
            yaxis=dict(showgrid=True, gridcolor="#ddd"),
            zaxis=dict(showgrid=True, gridcolor="#ddd"),
        ),
    )
    fig.show()
