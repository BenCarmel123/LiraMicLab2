"""Heatmap utilities for emotion similarity matrices.

Provides gamma boosting, plotting helpers, PNG export, and multi-page PDF export. Designed for similarity matrices (recommended range -1.0..1.0) with matching label lists.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


# ---------------------------------------------------------
# Optional enhancement: Gamma boosting
# ---------------------------------------------------------

def apply_gamma(matrix, gamma=2.0):
    """
    Boosts visual contrast without altering the original matrix values.
    """
    m_min, m_max = np.min(matrix), np.max(matrix)
    norm = (matrix - m_min) / (m_max - m_min + 1e-9)
    boosted = norm ** gamma
    return boosted * (m_max - m_min) + m_min


# ---------------------------------------------------------
# Core heatmap plotting
# ---------------------------------------------------------

def _plot_heatmap(ax, matrix, labels, title, gamma=2.0, suppress_diag=True):
    """
    Draws a high-contrast heatmap on a given axis.
    """

    # Visualization copy
    M = matrix.copy()

    # Reduce diagonal if desired (so 1.0 won't crush the color scale)
    if suppress_diag:
        np.fill_diagonal(M, 0.85)

    # Apply gamma boost
    M = apply_gamma(M, gamma=gamma)

    # Global color range for comparability
    im = ax.imshow(
        M,
        cmap="seismic",
        interpolation="nearest",
        vmin=-1.0,
        vmax=1.0
    )

    # Titles & labels
    ax.set_title(title)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    return im


# ---------------------------------------------------------
# Show heatmap interactively
# ---------------------------------------------------------

def show_heatmap(matrix, labels, title):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = _plot_heatmap(ax, matrix, labels, title)
    fig.colorbar(im)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------
# Save heatmap PNG
# ---------------------------------------------------------

def save_heatmap_png(matrix, labels, title, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = _plot_heatmap(ax, matrix, labels, title)
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# ---------------------------------------------------------
# Save all angles to a multi-page PDF
# ---------------------------------------------------------

def save_all_heatmaps_to_pdf(results_dict, pdf_path):
    """
    results_dict = {
        "front":  (matrix, labels),
        "left":   (matrix, labels),
        "right":  (matrix, labels)
    }
    """
    with PdfPages(pdf_path) as pdf:
        for angle, (matrix, labels) in results_dict.items():
            fig, ax = plt.subplots(figsize=(6, 5))
            im = _plot_heatmap(
                ax,
                matrix,
                labels,
                title=f"{angle.capitalize()} – Emotion Similarity Matrix"
            )
            fig.colorbar(im)
            plt.tight_layout()
            pdf.savefig()
            plt.close()

    print(f"[INFO] High-contrast PDF created: {pdf_path}")
