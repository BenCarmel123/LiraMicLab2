from build_matrix import build_matrix_from_parquet
from format_matrix import save_all_heatmaps_to_pdf

angles = ["front", "left", "right"]

BASE = "/Users/bencarmel/Documents/TAU/LiraMic/src/dataset/kdef_by_angle"

if __name__ == "__main__":
    results = {}

    for angle in angles:
        parquet_path = f"{BASE}/{angle}_embeddings.parquet"
        print(f"[INFO] Loading {parquet_path}")

        matrix, emotions = build_matrix_from_parquet(parquet_path)
        results[angle] = (matrix, emotions)

    pdf_path = f"{BASE}/emotion_similarity_matrices.pdf"
    save_all_heatmaps_to_pdf(results, pdf_path)
