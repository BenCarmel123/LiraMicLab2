"""
Pairwise Similarity Pipeline
----------------------------
For each angle (front/left/right):

1. Load all image embeddings
2. Filter + flatten embeddings
3. Normalize embeddings
4. Compute full NxN cosine similarity
5. Collapse to 7×7 emotion similarity matrix
6. Save all 7×7 matrices into a 3-page PDF

Uses shared utilities from similarity/utils.py
"""

import os
import sys
from tqdm import tqdm

# Make src importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from similarity.utils import (
    load_embeddings,
    normalize_embeddings,
    similarity_matrix,
    collapse_emotion_matrix
)

from heatmap.format_heatmap import save_all_heatmaps_to_pdf


# ---------------------------------------------------------
# MAIN PIPELINE FOR ONE ANGLE
# ---------------------------------------------------------

def compute_angle(angle_dir):
    """
    Runs the full pairwise similarity pipeline for a single angle.
    Returns: (7×7 matrix, emotion_labels)
    """

    print(f"[INFO] Loading embeddings from: {angle_dir}")
    E, emotions = load_embeddings(angle_dir)       # (N,D), list length N

    print(f"[INFO] Normalizing {E.shape[0]} embeddings...")
    E_norm = normalize_embeddings(E)

    print(f"[INFO] Computing full NxN similarity matrix...")
    sim_full = similarity_matrix(E_norm)

    print(f"[INFO] Collapsing to 7×7 emotion matrix...")
    mat, labels = collapse_emotion_matrix(sim_full, emotions)

    return mat, labels


# ---------------------------------------------------------
# PIPELINE ACROSS ALL ANGLES
# ---------------------------------------------------------

def run_pairwise_pipeline(base_path, angles):
    """
    base_path: directory containing subfolders front/, left/, right/
    Returns:
        results = {
            "front": (matrix, labels),
            "left":  (matrix, labels),
            "right": (matrix, labels)
        }
    """
    results = {}

    for angle in angles:
        print(f"\n[========== Processing: {angle} ==========]")

        angle_dir = os.path.join(base_path, angle)
        if not os.path.isdir(angle_dir):
            raise FileNotFoundError(f"Angle folder does not exist: {angle_dir}")

        results[angle] = compute_angle(angle_dir)

    print("\n[INFO] Pairwise similarity computation complete.")
    return results


# ---------------------------------------------------------
# EXECUTABLE SCRIPT
# ---------------------------------------------------------

if __name__ == "__main__":

    BASE = "/Users/bencarmel/Documents/TAU/LiraMic/src/dataset/kdef_by_angle"
    ANGLES = ["front", "left", "right"]

    print("[INFO] Running pairwise similarity pipeline...")
    results = run_pairwise_pipeline(BASE, ANGLES)

    pdf_path = f"{BASE}/emotion_similarity_pairwise.pdf"

    print(f"[INFO] Saving heatmaps to PDF: {pdf_path}")
    save_all_heatmaps_to_pdf(results, pdf_path)

    print("[DONE] PDF saved at:", pdf_path)
