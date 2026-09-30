import os
import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

def build_semantic_embeddings(
    dataset_path: str = "dataset_semantic.csv",
    model_name: str = "all-MiniLM-L6-v2",
    output_npy: str = "embeddings.npy",
    output_pkl: str = "embedder.pkl"
):
    print(f"[*] Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"[+] Loaded {len(df)} rows from {dataset_path}")

    # Ensure all option columns are string and handle NaNs
    for col in ["question", "opt_a", "opt_b", "opt_c", "opt_d"]:
        df[col] = df[col].fillna("").astype(str)

    # Build dense semantic text representation
    semantic_texts = (
        df["question"] + " " +
        "Option A: " + df["opt_a"] + " " +
        "Option B: " + df["opt_b"] + " " +
        "Option C: " + df["opt_c"] + " " +
        "Option D: " + df["opt_d"]
    ).str.strip().tolist()

    print(f"[*] Initializing SentenceTransformer model: {model_name}...")
    embedder = SentenceTransformer(model_name)

    print(f"[*] Encoding {len(semantic_texts)} questions into 384-d latent vectors (batch_size=32)...")
    embeddings = embedder.encode(
        semantic_texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    # Save artifacts
    np.save(output_npy, embeddings)
    joblib.dump(embedder, output_pkl)
    print(f"[+] Saved embeddings to {output_npy} with shape {embeddings.shape}")
    print(f"[+] Saved embedder model artifact to {output_pkl}")

    # Required output prints
    print("\n" + "=" * 70)
    print(f"EMBEDDINGS SHAPE: {embeddings.shape}")
    print("=" * 70)
    print("FIRST 20 DIMENSIONS OF QUESTION 0:")
    print(np.round(embeddings[0, :20], 4).tolist())
    print("=" * 70)
    print("3 SAMPLE QUESTIONS WITH FIRST 5 EMBEDDING VALUES:")
    for idx in [0, 1, 2]:
        q_text = df.iloc[idx]["question"]
        first_5 = np.round(embeddings[idx, :5], 4).tolist()
        print(f"Q[{idx}] '{q_text[:50]}...': {first_5}")
    print("=" * 70)

    return embeddings, embedder

if __name__ == "__main__":
    build_semantic_embeddings()
