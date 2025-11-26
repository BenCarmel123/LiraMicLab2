"""Compute and save face embeddings per angle.

build_angle_dataframe(angle_dir, label) returns a DataFrame with subject_id, emotion, angle, image_path, embedding. build_all_angle_embeddings(base_dir) generates parquet files for front/left/right and returns DataFrames. Requires model.hsem.get_embedding.
"""

import os
import pandas as pd
import numpy as np
from model.hsem import get_embedding
from tqdm import tqdm

# ------------------------------
# BUILD DATAFRAME FOR ONE ANGLE
# ------------------------------

def build_angle_dataframe(angle_dir, angle_label):
    """
    Creates a DataFrame for one angle:
       columns: subject_id, emotion, angle, image_path, embedding (list of floats)
    """
    rows = []

    for emotion in sorted(os.listdir(angle_dir)):
        emotion_dir = os.path.join(angle_dir, emotion)
        if not os.path.isdir(emotion_dir):
            continue

        for fname in tqdm(sorted(os.listdir(emotion_dir)), desc=f"Processing {angle_label}/{emotion}"):

            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(emotion_dir, fname)

            # subject ID = the prefix before underscore
            try:
                subject_id = int(fname.split("_")[0])
            except:
                subject_id = None

            emb = np.array(get_embedding(img_path)).squeeze()

            rows.append({
                "subject_id": subject_id,
                "emotion": emotion,
                "angle": angle_label,
                "image_path": img_path,
                "embedding": emb.tolist()        # <-- CRITICAL FIX
            })

    return pd.DataFrame(rows)


# ------------------------------
# BUILD ALL ANGLE DATAFRAMES
# ------------------------------

def build_all_angle_embeddings(base_dir):
    """
    Expected folder structure:

        base_dir/
            front/
            left/
            right/

    Saves:
        front_embeddings.parquet
        left_embeddings.parquet
        right_embeddings.parquet
    """

    angles = ["front", "left", "right"]
    embeddings_by_angle = {}

    for angle in angles:
        angle_dir = os.path.join(base_dir, angle)
        if not os.path.isdir(angle_dir):
            print(f"[WARN] Angle folder missing: {angle_dir}")
            continue

        print(f"[INFO] Processing angle: {angle}")

        df = build_angle_dataframe(angle_dir, angle)
        embeddings_by_angle[angle] = df

        out_path = os.path.join(base_dir, f"{angle}_embeddings.parquet")
        df.to_parquet(out_path)
        print(f"[INFO] Saved {out_path} with {len(df)} samples")

    return embeddings_by_angle


# ------------------------------
# MAIN
# ------------------------------

if __name__ == "__main__":
    BASE_DIR = "/Users/bencarmel/Documents/TAU/LiraMic/src/dataset/kdef_by_angle"
    build_all_angle_embeddings(BASE_DIR)
