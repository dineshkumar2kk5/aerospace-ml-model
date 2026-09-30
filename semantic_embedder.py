"""
semantic_embedder.py — Vectorize Aviation Meteorology Questions via Sentence Transformers
========================================================================================

Generates dense 384-dimensional semantic embeddings for all 806 DGCA exam questions
extracted from IC Joshi's "Aviation Meteorology" (7th Edition, 2023).

Model: 'all-MiniLM-L6-v2' (runs locally on CPU, ~80MB, zero recurring API cost).

Outputs:
    - embeddings.npy : NumPy array of shape (806, 384) containing normalized embeddings.
    - embedder.pkl   : Serialized SentenceTransformer instance for offline inference.

Usage:
    python semantic_embedder.py
    python semantic_embedder.py --csv dataset_semantic.csv --batch-size 32
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ── Windows Console UTF-8 Resilience ──────────────────────────────────────────
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ── Logging Setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("semantic_embedder")


def resolve_file_path(filename: str) -> Path:
    """
    Search for a file across the current directory, script directory, and ml/ directory.

    Args:
        filename: Name or relative path of the file.

    Returns:
        Resolved Path object.
    """
    candidates = [
        Path(filename),
        Path(__file__).resolve().parent / filename,
        Path(__file__).resolve().parent / "ml" / filename,
        Path(__file__).resolve().parent.parent / "ml" / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return Path(filename).resolve()


def build_semantic_text(df: pd.DataFrame) -> List[str]:
    """
    Concatenate question text and available multiple choice options into a unified
    dense semantic context string for embedding.

    Args:
        df: DataFrame containing 'question', 'opt_a', 'opt_b', 'opt_c', 'opt_d' columns.

    Returns:
        List of concatenated semantic text strings.
    """
    opt_cols = ["opt_a", "opt_b", "opt_c", "opt_d"]
    for col in ["question"] + opt_cols:
        if col not in df.columns:
            df[col] = ""

    # Gracefully fill NaN with empty strings
    clean_q = df["question"].fillna("").astype(str).str.strip()
    clean_a = df["opt_a"].fillna("").astype(str).str.strip()
    clean_b = df["opt_b"].fillna("").astype(str).str.strip()
    clean_c = df["opt_c"].fillna("").astype(str).str.strip()
    clean_d = df["opt_d"].fillna("").astype(str).str.strip()

    semantic_strings = []
    for q, a, b, c, d in zip(clean_q, clean_a, clean_b, clean_c, clean_d):
        tokens = [q]
        if a:
            tokens.append(f"A: {a}")
        if b:
            tokens.append(f"B: {b}")
        if c:
            tokens.append(f"C: {c}")
        if d:
            tokens.append(f"D: {d}")
        semantic_strings.append(" ".join(tokens).strip())

    return semantic_strings


def generate_and_save_embeddings(
    csv_path: str = "dataset_semantic.csv",
    model_name: str = "all-MiniLM-L6-v2",
    output_embeddings: str = "embeddings.npy",
    output_model: str = "embedder.pkl",
    batch_size: int = 32,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, SentenceTransformer]:
    """
    Load question dataset, encode semantic context into 384-dim vectors, and persist artifacts.

    Args:
        csv_path: Path to dataset_semantic.csv.
        model_name: SentenceTransformer pretrained model tag.
        output_embeddings: Filename for output numpy array.
        output_model: Filename for serialized SentenceTransformer.
        batch_size: Batch size for GPU/CPU vectorized inference.
        device: 'cpu' or 'cuda'. Defaults to auto-selection with CPU fallback.

    Returns:
        Tuple of (embeddings_array, model_instance).
    """
    resolved_csv = resolve_file_path(csv_path)
    logger.info(f"🔍 Loading dataset from: {resolved_csv}")
    if not resolved_csv.exists():
        raise FileNotFoundError(f"❌ Dataset file not found at: {resolved_csv}")

    df = pd.read_csv(resolved_csv)
    logger.info(f"📊 Loaded {len(df)} questions from {resolved_csv.name}")

    # Build semantic text representation
    texts = build_semantic_text(df)

    # Automatically select CPU or CUDA safely
    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    logger.info(f"🚀 Initializing SentenceTransformer('{model_name}') on device='{device}'...")
    model = SentenceTransformer(model_name, device=device)

    logger.info(f"🧠 Encoding {len(texts)} questions (batch_size={batch_size}, normalize=True)...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    # Verify array structure
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    shape = embeddings.shape
    logger.info(f"✅ Embeddings generated with shape: {shape}")

    # Save embeddings to disk
    out_emb_path = Path(output_embeddings).resolve()
    np.save(out_emb_path, embeddings)
    logger.info(f"💾 Saved embeddings array to: {out_emb_path}")

    # Also mirror to ml/ directory if it exists for backwards-compatibility
    ml_emb = Path(__file__).resolve().parent / "ml" / "embeddings.npy"
    if ml_emb.parent.exists() and ml_emb != out_emb_path:
        np.save(ml_emb, embeddings)

    # Save embedder model instance via joblib
    out_model_path = Path(output_model).resolve()
    joblib.dump(model, out_model_path)
    logger.info(f"💾 Saved embedder model instance to: {out_model_path}")

    ml_model = Path(__file__).resolve().parent / "ml" / "embedder.pkl"
    if ml_model.parent.exists() and ml_model != out_model_path:
        joblib.dump(model, ml_model)

    return embeddings, model


def display_embedding_inspection(
    df: pd.DataFrame, embeddings: np.ndarray, num_samples: int = 3
) -> None:
    """
    Print proof of semantic vectorization: shape, vector slice, and sample embeddings.

    Args:
        df: DataFrame of questions.
        embeddings: NumPy embeddings matrix.
        num_samples: Number of sample questions to inspect.
    """
    print("\n" + "═" * 70)
    print("📊 SEMANTIC EMBEDDING VERIFICATION REPORT")
    print("═" * 70)
    print(f"✅ Final Embedding Matrix Shape : {embeddings.shape} (N_questions × Dims)")
    print(f"✅ Data Type                    : {embeddings.dtype}")
    print(f"✅ L2-Norm of Row 0             : {np.linalg.norm(embeddings[0]):.4f} (Normalized Unit Vector)")
    print("\n📐 First 20 Dimensions of Question #1 Vector:")
    print("   " + ", ".join(f"{val:+.4f}" for val in embeddings[0][:20]))
    print("\n" + "─" * 70)
    print("🔍 Sample Questions & First 5 Embedding Dimensions:")
    print("─" * 70)

    for i in range(min(num_samples, len(df))):
        q_text = df.iloc[i].get("question", "N/A")
        topic = df.iloc[i].get("topic", "N/A")
        diff = df.iloc[i].get("difficulty", "N/A")
        first_5 = [f"{v:+.4f}" for v in embeddings[i][:5]]
        print(f" [{i+1}] Topic: {topic} | Difficulty: {diff}")
        print(f"     Q: \"{q_text[:75]}...\"")
        print(f"     Vector[:5] -> [{', '.join(first_5)}]")
        print()
    print("═" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 384-dim semantic embeddings for DGCA questions.")
    parser.add_argument("--csv", default="dataset_semantic.csv", help="Input CSV path")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Pretrained model identifier")
    parser.add_argument("--batch-size", type=int, default=32, help="Encoding batch size")
    parser.add_argument("--output-emb", default="embeddings.npy", help="Output .npy filepath")
    parser.add_argument("--output-model", default="embedder.pkl", help="Output model pickle filepath")
    args = parser.parse_args()

    emb, mdl = generate_and_save_embeddings(
        csv_path=args.csv,
        model_name=args.model,
        output_embeddings=args.output_emb,
        output_model=args.output_model,
        batch_size=args.batch_size,
    )

    resolved_csv = resolve_file_path(args.csv)
    dataset_df = pd.read_csv(resolved_csv)
    display_embedding_inspection(dataset_df, emb, num_samples=3)
