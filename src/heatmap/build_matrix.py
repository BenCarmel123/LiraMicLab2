import numpy as np
import pandas as pd

# ---------------------------------------------------
# 1. COMPUTE MEAN EMBEDDING PER EMOTION (PROTOTYPE)
# ---------------------------------------------------

def compute_prototypes(df):
    prototypes = {}
    emotions = sorted(df["emotion"].unique())
    for emotion in emotions:
        sub_df = df[df["emotion"] == emotion]

        # Convert each embedding to a flat (1280,) vector
        flat_embs = [np.array(e).reshape(-1) for e in sub_df["embedding"].values]

        emb_matrix = np.vstack(flat_embs)   # shape = (N, 1280)

        prototypes[emotion] = emb_matrix.mean(axis=0)  # shape (1280,)

    return prototypes


# ---------------------------------------------------
# 2. COSINE SIMILARITY BETWEEN TWO VECTORS
# ---------------------------------------------------

def cosine_sim(vec1, vec2):
    # Convert to numpy arrays and force flattening of nested vectors
    v1 = np.asarray(vec1)
    v2 = np.asarray(vec2)

    # If embedding wrapped inside object array → unwrap it
    if v1.dtype == object:
        v1 = np.concatenate(v1).reshape(-1)
    else:
        v1 = v1.reshape(-1)

    if v2.dtype == object:
        v2 = np.concatenate(v2).reshape(-1)
    else:
        v2 = v2.reshape(-1)

    # Compute norms
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))

    if n1 == 0 or n2 == 0:
        return 0.0
    res = float(np.dot(v1, v2) / (n1 * n2))
    return res

# ---------------------------------------------------
# 3. BUILD A 7x7 MATRIX FROM PROTOTYPES
# ---------------------------------------------------

def prototypes_to_matrix(prototypes):
    """
    Given:
        prototypes = { "angry": v1, "happy": v2, ... }

    Returns:
        similarity_matrix (N x N numpy array)
        emotions_order (list of emotion names)
    """
    emotions = sorted(prototypes.keys())
    n = len(emotions)

    matrix = np.zeros((n, n))

    for i, e1 in enumerate(emotions):
        for j, e2 in enumerate(emotions):
            matrix[i, j] = cosine_sim(prototypes[e1], prototypes[e2])

    return matrix, emotions


# ---------------------------------------------------
# OPTIONAL: Utility function to run on a parquet file
# ---------------------------------------------------

def build_matrix_from_parquet(parquet_path):
    """
    Loads a parquet file for an angle and returns:
        similarity_matrix, emotions_order
    """
    df = pd.read_parquet(parquet_path)
    prototypes = compute_prototypes(df)
    matrix, emotions = prototypes_to_matrix(prototypes)
    return matrix, emotions
