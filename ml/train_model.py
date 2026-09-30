"""
train_model.py — Model Training Pipeline for AeroBeacon Difficulty Prediction
==============================================================================

Trains and evaluates multiple classifiers on the synthetic (or real) question
dataset, tunes the best performer via GridSearchCV, and persists the production
model along with metadata.

Artifacts produced:
    - difficulty_model.pkl      : Trained RandomForest (or best) classifier
    - label_encoder.pkl         : Scikit-learn LabelEncoder for difficulty labels
    - model_metadata.json       : Accuracy, params, timestamp, feature names
    - model_comparison.png      : Bar chart of model accuracies
    - feature_importance.png    : Horizontal bar chart of feature importances
    - confusion_matrix.png      : Heatmap of the tuned model confusion matrix

Usage:
    python train_model.py                       # defaults
    python train_model.py --dataset data.csv    # custom CSV
    python train_model.py --output-dir models/  # custom output
"""

import argparse
import json
import logging
import os
import sys

# Ensure non-interactive backend before ANY matplotlib or sub-imports
os.environ["MPLBACKEND"] = "Agg"

from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
FEATURE_COLUMNS = ["text_length", "num_options", "avg_time_taken", "past_accuracy"]
LABEL_COLUMN = "difficulty"
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

# Hyperparameter grid for RandomForest tuning
RF_PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [6, 10, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
}


def load_dataset(path: str) -> pd.DataFrame:
    """
    Load and validate the CSV dataset.

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Validated dataframe ready for training.

    Raises
    ------
    FileNotFoundError
        If the CSV file doesn't exist.
    ValueError
        If required columns are missing.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path.resolve()}")

    df = pd.read_csv(csv_path)
    logger.info("Loaded dataset: %d rows × %d columns from %s", *df.shape, csv_path)

    # Validate required columns
    required = set(FEATURE_COLUMNS + [LABEL_COLUMN])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    # Check for NaN in features
    nan_counts = df[FEATURE_COLUMNS].isna().sum()
    if nan_counts.any():
        logger.warning("NaN values detected in features:\n%s", nan_counts[nan_counts > 0])
        logger.info("Dropping %d rows with NaN values.", df[FEATURE_COLUMNS].isna().any(axis=1).sum())
        df = df.dropna(subset=FEATURE_COLUMNS)

    # Validate difficulty labels
    valid_labels = {"Easy", "Medium", "Hard"}
    actual_labels = set(df[LABEL_COLUMN].unique())
    invalid = actual_labels - valid_labels
    if invalid:
        raise ValueError(f"Invalid difficulty labels found: {invalid}")

    logger.info("Label distribution:\n%s", df[LABEL_COLUMN].value_counts().to_string())
    return df


def train_and_compare(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_full: pd.DataFrame,
    y_full: pd.Series,
) -> dict:
    """
    Train multiple classifiers and compare their performance.

    Returns
    -------
    dict
        Mapping of model name → (accuracy, cv_score, trained_model).
    """
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }

    results = {}
    logger.info("─" * 60)
    logger.info("%-22s | %10s | %10s", "Model", "Test Acc", "CV (5-fold)")
    logger.info("─" * 60)

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        cv = cross_val_score(model, X_full, y_full, cv=CV_FOLDS, n_jobs=1).mean()
        results[name] = {"accuracy": acc, "cv_score": cv, "model": model}
        logger.info("%-22s | %10.4f | %10.4f", name, acc, cv)

    logger.info("─" * 60)
    return results


def plot_comparison(results: dict, output_dir: Path) -> None:
    """Save a bar chart comparing model accuracies."""
    names = list(results.keys())
    test_acc = [results[n]["accuracy"] for n in names]
    cv_acc = [results[n]["cv_score"] for n in names]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, test_acc, width, label="Test Accuracy", color="#1b5e20")
    bars2 = ax.bar(x + width / 2, cv_acc, width, label="CV Score", color="#66bb6a")

    ax.set_ylabel("Score")
    ax.set_title("AeroBeacon — Model Accuracy Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar_group in [bars1, bars2]:
        for bar in bar_group:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout()
    path = output_dir / "model_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("📊 Saved model comparison chart → %s", path)


def tune_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    le: LabelEncoder,
) -> RandomForestClassifier:
    """
    Hyperparameter-tune a RandomForest via GridSearchCV.

    Returns
    -------
    RandomForestClassifier
        The best estimator found by grid search.
    """
    logger.info("🔧 Tuning RandomForest with GridSearchCV (%d-fold)...", CV_FOLDS)
    grid = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        RF_PARAM_GRID,
        cv=CV_FOLDS,
        n_jobs=1,
        scoring="accuracy",
        verbose=0,
    )
    grid.fit(X_train, y_train)
    best = grid.best_estimator_

    pred = best.predict(X_test)
    acc = accuracy_score(y_test, pred)

    logger.info("Best params  : %s", grid.best_params_)
    logger.info("Tuned accuracy: %.4f", acc)
    logger.info(
        "\nClassification Report:\n%s",
        classification_report(y_test, pred, target_names=le.classes_),
    )

    return best


def plot_feature_importance(model, feature_names: list, output_dir: Path) -> None:
    """Save a horizontal bar chart of feature importances."""
    importances = model.feature_importances_
    indices = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        [feature_names[i] for i in indices],
        importances[indices],
        color="#2e7d32",
    )
    ax.set_xlabel("Importance")
    ax.set_title("AeroBeacon — Feature Importance (Random Forest)")
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    path = output_dir / "feature_importance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("📊 Saved feature importance chart → %s", path)


def plot_confusion_matrix(
    model, X_test, y_test, le: LabelEncoder, output_dir: Path
) -> None:
    """Save a confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test,
        display_labels=le.classes_,
        cmap="Greens",
        ax=ax,
    )
    ax.set_title("AeroBeacon — Confusion Matrix (Tuned Model)")

    fig.tight_layout()
    path = output_dir / "confusion_matrix.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("📊 Saved confusion matrix → %s", path)


def save_artifacts(
    model,
    le: LabelEncoder,
    accuracy: float,
    best_params: dict,
    output_dir: Path,
) -> None:
    """Persist model, encoder, and metadata to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "difficulty_model.pkl"
    le_path = output_dir / "label_encoder.pkl"
    meta_path = output_dir / "model_metadata.json"

    joblib.dump(model, model_path)
    joblib.dump(le, le_path)

    metadata = {
        "model_type": type(model).__name__,
        "accuracy": round(accuracy, 4),
        "best_params": best_params,
        "features": FEATURE_COLUMNS,
        "labels": list(le.classes_),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": pd.__version__,  # pandas version as proxy
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "cv_folds": CV_FOLDS,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("✅ Saved model      → %s", model_path)
    logger.info("✅ Saved encoder     → %s", le_path)
    logger.info("✅ Saved metadata    → %s", meta_path)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train the AeroBeacon difficulty prediction model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset", type=str, default="dataset.csv",
        help="Path to the training CSV dataset",
    )
    parser.add_argument(
        "--output-dir", type=str, default=".",
        help="Directory to save model artifacts",
    )
    return parser.parse_args()


def main() -> None:
    """Full training pipeline entry point."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load dataset
    df = load_dataset(args.dataset)

    # 2. Encode labels
    le = LabelEncoder()
    df["label"] = le.fit_transform(df[LABEL_COLUMN])

    X = df[FEATURE_COLUMNS]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )
    logger.info("Train/Test split: %d / %d", len(X_train), len(X_test))

    # 3. Compare models
    results = train_and_compare(X_train, y_train, X_test, y_test, X, y)
    plot_comparison(results, output_dir)

    # 4. Identify best model
    best_name = max(results, key=lambda k: results[k]["accuracy"])
    logger.info("🏆 Best base model: %s (%.4f)", best_name, results[best_name]["accuracy"])

    # 5. Tune RandomForest (our production model)
    final_model = tune_random_forest(X_train, y_train, X_test, y_test, le)
    final_acc = accuracy_score(y_test, final_model.predict(X_test))

    # 6. Visualizations
    plot_feature_importance(final_model, list(X.columns), output_dir)
    plot_confusion_matrix(final_model, X_test, y_test, le, output_dir)

    # 7. Save artifacts
    save_artifacts(
        model=final_model,
        le=le,
        accuracy=final_acc,
        best_params=final_model.get_params(),
        output_dir=output_dir,
    )

    logger.info("🎉 Training pipeline complete!")


if __name__ == "__main__":
    main()
