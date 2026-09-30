"""
hybrid_train.py — Train Hybrid (Semantic 384-Dim + Structural 4-Dim) Classifiers
=================================================================================

Constructs a 388-dimensional hybrid feature matrix combining:
  1. Dense semantic sentence embeddings (384 dims from 'all-MiniLM-L6-v2')
  2. Structural metadata features (4 dims: text_length, num_options, avg_time_taken, past_accuracy)

Evaluates 5 distinct machine learning architectures via Stratified 5-Fold Cross-Validation,
generates publication-grade evaluation visuals (model comparison, confusion matrix, t-SNE plot),
and serializes the best performing model as `hybrid_semantic_model.pkl`.

Outputs:
  - hybrid_model_comparison.png : Test Accuracy vs 5-Fold Stratified CV bar chart.
  - hybrid_confusion.png        : Confusion matrix heatmap for the best classifier.
  - semantic_tsne.png           : 2D t-SNE manifold visualization colored by difficulty.
  - hybrid_semantic_model.pkl   : Best serialized classifier.
  - hybrid_feature_names.pkl    : Feature names list for traceability.

Usage:
  python hybrid_train.py
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Headless matplotlib backend before any pyplot import
os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

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
logger = logging.getLogger("hybrid_train")


def resolve_file_path(filename: str) -> Path:
    """
    Search for a file across the current directory, script directory, and ml/ directory.
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


def load_data_and_embeddings(
    csv_path: str = "dataset_semantic.csv",
    emb_path: str = "embeddings.npy",
    le_path: str = "label_encoder.pkl",
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, LabelEncoder, List[str]]:
    """
    Load tabular metadata, embeddings, and existing label encoder. Assemble 388-dim matrix.

    Returns:
        X_hybrid: Matrix of shape (806, 388)
        y: Target label indices array of shape (806,)
        df: Raw DataFrame
        label_encoder: Fitted LabelEncoder
        feature_names: List of all 388 feature identifiers
    """
    res_csv = resolve_file_path(csv_path)
    res_emb = resolve_file_path(emb_path)
    res_le = resolve_file_path(le_path)

    if not res_csv.exists():
        raise FileNotFoundError(f"❌ Missing dataset CSV: {res_csv}")
    if not res_emb.exists():
        raise FileNotFoundError(f"❌ Missing embeddings file: {res_emb}. Run semantic_embedder.py first!")
    if not res_le.exists():
        raise FileNotFoundError(f"❌ Missing label encoder: {res_le}")

    logger.info(f"📂 Loading dataset from: {res_csv}")
    df = pd.read_csv(res_csv)

    logger.info(f"📂 Loading embeddings from: {res_emb}")
    embeddings = np.load(res_emb)

    logger.info(f"📂 Loading existing label encoder from: {res_le}")
    label_encoder = joblib.load(res_le)

    # Validate structural columns
    structural_cols = ["text_length", "num_options", "avg_time_taken", "past_accuracy"]
    for col in structural_cols:
        if col not in df.columns:
            if col == "text_length":
                df[col] = df["question"].fillna("").astype(str).str.len()
            elif col == "num_options":
                df[col] = 4
            elif col == "avg_time_taken":
                df[col] = 45.0
            elif col == "past_accuracy":
                df[col] = 0.75

    structural_features = df[structural_cols].fillna(df[structural_cols].mean()).values

    # Concatenate 384 semantic dims + 4 structural dims -> 388 dims
    X_hybrid = np.hstack([embeddings, structural_features]).astype(np.float32)

    # Encode target labels using existing label encoder
    if "difficulty" in df.columns:
        y = label_encoder.transform(df["difficulty"].astype(str))
    elif "label" in df.columns:
        y = df["label"].values.astype(int)
    else:
        raise ValueError("❌ Target column 'difficulty' or 'label' not found in dataset.")

    # Create descriptive feature names
    semantic_names = [f"emb_dim_{i:03d}" for i in range(embeddings.shape[1])]
    feature_names = semantic_names + structural_cols

    logger.info(f"✅ Assembled Hybrid Feature Matrix: shape={X_hybrid.shape} (384 semantic + 4 structural)")
    logger.info(f"✅ Target distribution: {dict(pd.Series(y).value_counts())}")

    return X_hybrid, y, df, label_encoder, feature_names


def train_and_evaluate_models(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Dict]:
    """
    Train 5 distinct classifiers and assess both Test Accuracy and 5-Fold Stratified CV.
    """
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, C=1.0, random_state=42),
        "Support Vector Machine": SVC(kernel="rbf", probability=True, C=1.0, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
        "Multi-Layer Perceptron": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=800, random_state=42),
    }

    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    logger.info("🚀 Training 5 candidate classifiers on 388-dim hybrid features...")
    for name, clf in models.items():
        logger.info(f"   ⚙️ Training {name}...")
        clf.fit(X_train, y_train)

        # Predictions on held-out test split
        y_pred = clf.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)

        # 5-Fold Cross-Validation on training set
        cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
        mean_cv = float(np.mean(cv_scores))
        std_cv = float(np.std(cv_scores))

        results[name] = {
            "model": clf,
            "test_accuracy": float(test_acc),
            "cv_mean": mean_cv,
            "cv_std": std_cv,
            "y_pred": y_pred,
        }
        logger.info(f"      Test Acc: {test_acc*100:.2f}% | 5-Fold CV: {mean_cv*100:.2f}% (±{std_cv*100:.2f}%)")

    return results


def plot_model_comparison(results: Dict[str, Dict], output_path: str = "hybrid_model_comparison.png") -> None:
    """
    Plot side-by-side grouped bar chart comparing Test Accuracy and 5-Fold CV.
    """
    model_names = list(results.keys())
    test_accs = [results[m]["test_accuracy"] * 100 for m in model_names]
    cv_accs = [results[m]["cv_mean"] * 100 for m in model_names]
    cv_errs = [results[m]["cv_std"] * 100 for m in model_names]

    x = np.arange(len(model_names))
    width = 0.35

    plt.figure(figsize=(10, 5.5), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    rects1 = plt.bar(x - width / 2, test_accs, width, label="Test Accuracy (20%)", color="#2563EB", alpha=0.9)
    rects2 = plt.bar(x + width / 2, cv_accs, width, yerr=cv_errs, capsize=5, label="5-Fold Stratified CV", color="#10B981", alpha=0.9)

    plt.ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    plt.title("AeroBeacon Hybrid Model Benchmark (388-Dim Semantic + Structural)", fontsize=13, fontweight="bold", pad=15)
    plt.xticks(x, model_names, fontsize=10, rotation=15, ha="right")
    plt.ylim(0, 105)
    plt.legend(frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Attach value labels above bars
    for rect in rects1:
        h = rect.get_height()
        plt.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    for rect in rects2:
        h = rect.get_height()
        plt.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"📊 Saved model comparison chart to: {output_path}")


def plot_confusion_matrix(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_path: str = "hybrid_confusion.png",
) -> None:
    """
    Render and save confusion matrix heatmap.
    """
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6.5, 5.5), dpi=300)
    try:
        import seaborn as sns
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            cbar=False,
            annot_kws={"size": 13, "weight": "bold"},
        )
    except ImportError:
        # Graceful fallback to pure matplotlib
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, fontsize=11)
        plt.yticks(tick_marks, class_names, fontsize=11)
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color="white" if cm[i, j] > thresh else "black", fontsize=13, fontweight="bold")

    plt.title("Confusion Matrix — Best Hybrid Model", fontsize=13, fontweight="bold", pad=12)
    plt.ylabel("True Difficulty Label", fontsize=11, fontweight="bold")
    plt.xlabel("Predicted Difficulty Label", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"📊 Saved confusion matrix heatmap to: {output_path}")


def compute_and_plot_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    output_path: str = "semantic_tsne.png",
) -> None:
    """
    Compute 2D t-SNE projection of 384-dim semantic embeddings and plot clusters colored by difficulty.
    """
    logger.info("🧠 Computing 2D t-SNE projection of 384-dimensional question embeddings...")
    try:
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca", max_iter=1000)
        emb_2d = tsne.fit_transform(embeddings)
    except (TypeError, ValueError):
        # Fallback for sklearn versions with different parameter support
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="random")
        emb_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8.5, 6.5), dpi=300)
    colors = {"Easy": "#10B981", "Medium": "#F59E0B", "Hard": "#EF4444"}

    for idx, cname in enumerate(class_names):
        mask = labels == idx
        color = colors.get(cname, "#6B7280")
        plt.scatter(
            emb_2d[mask, 0],
            emb_2d[mask, 1],
            c=color,
            label=f"{cname} ({np.sum(mask)} questions)",
            alpha=0.75,
            edgecolors="none",
            s=45,
        )

    plt.title("t-SNE Manifold of 384-Dim Semantic Embeddings\n(IC Joshi Aviation Meteorology Question Bank)", fontsize=13, fontweight="bold", pad=14)
    plt.xlabel("t-SNE Dimension 1", fontsize=11, fontweight="bold")
    plt.ylabel("t-SNE Dimension 2", fontsize=11, fontweight="bold")
    plt.legend(frameon=True, loc="upper right", facecolor="#F8FAFC", edgecolor="#CBD5E1", title="Difficulty Tier")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"📊 Saved semantic t-SNE scatter plot to: {output_path}")


def run_pipeline() -> None:
    """
    Execute full hybrid training, benchmarking, chart generation, and model persistence.
    """
    X_hybrid, y, df, label_encoder, feature_names = load_data_and_embeddings()

    # Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_hybrid, y, test_size=0.20, random_state=42, stratify=y
    )
    logger.info(f"📊 Train shape: {X_train.shape} | Test shape: {X_test.shape}")

    # Benchmark all 5 models
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    # Plot comparative model performance
    plot_model_comparison(results, output_path="hybrid_model_comparison.png")

    # Select best model based on combined test accuracy and CV
    best_name = max(results.keys(), key=lambda k: (results[k]["cv_mean"] + results[k]["test_accuracy"]))
    best_info = results[best_name]
    best_model = best_info["model"]
    logger.info(f"\n🏆 Best Performing Hybrid Architecture: {best_name}")
    logger.info(f"   Test Accuracy: {best_info['test_accuracy']*100:.2f}% | 5-Fold CV: {best_info['cv_mean']*100:.2f}%")

    # Confusion matrix & classification report for best model
    class_names = list(label_encoder.classes_)
    plot_confusion_matrix(y_test, best_info["y_pred"], class_names, output_path="hybrid_confusion.png")

    print("\n" + "═" * 70)
    print(f"📊 CLASSIFICATION REPORT — BEST HYBRID MODEL ({best_name})")
    print("═" * 70)
    print(classification_report(y_test, best_info["y_pred"], target_names=class_names, digits=4))
    print("═" * 70)

    # 2D t-SNE plot using raw 384 semantic embeddings
    embeddings = X_hybrid[:, :384]
    compute_and_plot_tsne(embeddings, y, class_names, output_path="semantic_tsne.png")

    # Save best hybrid model and feature names
    joblib.dump(best_model, "hybrid_semantic_model.pkl")
    joblib.dump(feature_names, "hybrid_feature_names.pkl")
    logger.info("💾 Saved best hybrid model to: hybrid_semantic_model.pkl")
    logger.info("💾 Saved feature names to: hybrid_feature_names.pkl")

    # Mirror artifacts to ml/ directory if it exists
    ml_dir = Path(__file__).resolve().parent / "ml"
    if ml_dir.exists():
        joblib.dump(best_model, ml_dir / "hybrid_semantic_model.pkl")
        joblib.dump(feature_names, ml_dir / "hybrid_feature_names.pkl")


if __name__ == "__main__":
    run_pipeline()
