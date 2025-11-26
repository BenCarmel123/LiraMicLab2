import os
import numpy as np
from tqdm import tqdm
from model.hsem import get_embedding


# ---------------------------------------------------------
# CLEAN ONE EMBEDDING VECTOR
# ---------------------------------------------------------

def clean_embedding(vec):
    """
    Takes a raw embedding and returns a clean 1D float32 vector.
    Handles:
        - None
        - object arrays
        - nested lists/arrays
        - incorrect shapes
        - zero vectors
    Returns:
        np.ndarray shape (D,) or None if corrupted
    """
    if vec is None:
        return None

    v = np.asarray(vec)

    # unwrap object-array nesting
    if v.dtype == object:
        try:
            v = np.concatenate(v)
        except:
            return None

    try:
        v = v.reshape(-1)
    except:
        return None

    # corrupted or wrong dimension
    if v.size < 100:
        return None

    # zero vector = useless
    if np.all(v == 0):
        return None

    return v.astype(np.float32)


# ---------------------------------------------------------
# LOAD ALL EMBEDDINGS FOR AN ANGLE (front/left/right)
# ---------------------------------------------------------

def load_embeddings(angle_dir):
    """
    Loads + cleans embeddings for one angle.
    
    Returns:
        E : ndarray (N, D)
        emotions : list[str] length N

    Also prints corruption stats by emotion.
    """
    all_embeddings = []
    all_emotions = []

    corruption_counts = {}
    total_counts = {}

    # iterate emotion folders
    for emotion in sorted(os.listdir(angle_dir)):
        emotion_dir = os.path.join(angle_dir, emotion)
        if not os.path.isdir(emotion_dir):
            continue

        files = sorted([
            f for f in os.listdir(emotion_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        total_counts[emotion] = len(files)
        corruption_counts[emotion] = 0

        print(f"  >> Emotion '{emotion}' — {len(files)} images")

        # tqdm loop
        for fname in tqdm(files, desc=f"{emotion:>8}", leave=False):
            path = os.path.join(emotion_dir, fname)

            # compute embedding
            try:
                raw = get_embedding(path)
            except:
                corruption_counts[emotion] += 1
                continue

            # clean it
            emb = clean_embedding(raw)
            if emb is None:
                corruption_counts[emotion] += 1
                continue

            all_embeddings.append(emb)
            all_emotions.append(emotion)

    # -----------------------------
    # Corruption summary
    # -----------------------------
    print("\n[EMBEDDING CLEANING SUMMARY]")
    total_corrupted = 0

    for emo in corruption_counts:
        c = corruption_counts[emo]
        t = total_counts[emo]
        total_corrupted += c
        pct = (c / t) if t > 0 else 0
        print(f"  {emo:>9}: {c}/{t} corrupted ({pct:.1%})")

    print(f"\n[INFO] TOTAL CORRUPTED ACROSS ANGLE = {total_corrupted}\n")

    # -----------------------------
    # Convert lists → ndarray
    # -----------------------------
    if len(all_embeddings) == 0:
        raise ValueError(f"No valid embeddings found in {angle_dir}")

    E = np.vstack(all_embeddings)  # (N, D)
    return E, all_emotions


# ---------------------------------------------------------
# NORMALIZATION (fast)
# ---------------------------------------------------------

def normalize_embeddings(E):
    """
    E: (N, D)
    Returns normalized embeddings.
    """
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    return E / norms


# ---------------------------------------------------------
# FULL NxN SIMILARITY MATRIX USING COSINE
# ---------------------------------------------------------

def similarity_matrix(E):
    """
    E: normalized embeddings (N, D)
    Returns NxN cosine similarity matrix.
    """
    E_norm = normalize_embeddings(E)
    return E_norm @ E_norm.T   # (N, N)


# ---------------------------------------------------------
# REDUCE NxN → 7×7 (emotion × emotion)
# ---------------------------------------------------------

def collapse_emotion_matrix(sim_M, emotions):
    """
    sim_M: (N × N)
    emotions: list[str], length N

    Returns:
        mat7 : (7×7)
        unique_emotions : list[str]
    """
    emotions = np.array(emotions)
    unique = sorted(list(set(emotions)))
    k = len(unique)

    mat = np.zeros((k, k))

    # index lists for each emotion
    idx = {e: np.where(emotions == e)[0] for e in unique}

    for i, e1 in enumerate(unique):
        for j, e2 in enumerate(unique):
            sub = sim_M[np.ix_(idx[e1], idx[e2])]
            mat[i, j] = sub.mean()

    return mat, unique
