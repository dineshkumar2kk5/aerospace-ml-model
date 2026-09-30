"""
generate_dataset.py — Synthetic Dataset Generator for AeroBeacon ML Pipeline
=============================================================================

Generates a labeled CSV dataset of aviation exam questions with features that
correlate to difficulty. Used to bootstrap the ML model before real student
attempt data is available.

Features generated:
    - text_length     : Proxy for question complexity (characters)
    - num_options     : Number of MCQ choices (always 4 for DGCA exams)
    - avg_time_taken  : Average seconds students spend on similar questions
    - past_accuracy   : Historical correct-answer rate (0.0–1.0)
    - topic           : DGCA syllabus topic category
    - difficulty       : Ground-truth label (Easy / Medium / Hard)

Distribution follows the target exam ratio: 30% Easy / 50% Medium / 20% Hard.

Usage:
    python generate_dataset.py                   # default 500 rows
    python generate_dataset.py --rows 1000       # custom size
    python generate_dataset.py --output data.csv # custom filename
    python generate_dataset.py --seed 123        # reproducibility
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
TOPICS = [
    "Meteorology",
    "Navigation",
    "Air Regulations",
    "Technical General",
    "Radio Aids",
    "Human Performance",
    "Flight Planning",
    "Aircraft Systems",
]

# Difficulty distribution ratios (must sum to 1.0)
DIFFICULTY_RATIOS = {"Easy": 0.30, "Medium": 0.50, "Hard": 0.20}

# Feature ranges per difficulty — (min, max) inclusive
FEATURE_RANGES = {
    "Easy": {
        "text_length": (80, 150),
        "avg_time_taken": (20, 45),
        "past_accuracy": (0.70, 0.95),
    },
    "Medium": {
        "text_length": (150, 250),
        "avg_time_taken": (45, 75),
        "past_accuracy": (0.40, 0.70),
    },
    "Hard": {
        "text_length": (220, 350),
        "avg_time_taken": (75, 130),
        "past_accuracy": (0.10, 0.40),
    },
}

# Topic weights — some topics are harder on average (adds realism)
TOPIC_WEIGHTS = {
    "Meteorology": 0.15,
    "Navigation": 0.15,
    "Air Regulations": 0.20,
    "Technical General": 0.10,
    "Radio Aids": 0.10,
    "Human Performance": 0.10,
    "Flight Planning": 0.10,
    "Aircraft Systems": 0.10,
}


def validate_ratios() -> None:
    """Ensure difficulty ratios sum to 1.0 (within floating-point tolerance)."""
    total = sum(DIFFICULTY_RATIOS.values())
    if not np.isclose(total, 1.0):
        raise ValueError(
            f"DIFFICULTY_RATIOS must sum to 1.0, got {total:.4f}"
        )


def validate_topic_weights() -> None:
    """Ensure topic weights sum to 1.0."""
    total = sum(TOPIC_WEIGHTS.values())
    if not np.isclose(total, 1.0):
        raise ValueError(
            f"TOPIC_WEIGHTS must sum to 1.0, got {total:.4f}"
        )


def generate_row(difficulty: str, rng: np.random.Generator) -> dict:
    """
    Generate a single synthetic question feature row.

    Parameters
    ----------
    difficulty : str
        One of 'Easy', 'Medium', 'Hard'.
    rng : numpy.random.Generator
        Seeded random number generator for reproducibility.

    Returns
    -------
    dict
        Feature dictionary for one question row.

    Raises
    ------
    ValueError
        If difficulty is not a recognized label.
    """
    if difficulty not in FEATURE_RANGES:
        raise ValueError(
            f"Unknown difficulty '{difficulty}'. "
            f"Expected one of {list(FEATURE_RANGES.keys())}"
        )

    ranges = FEATURE_RANGES[difficulty]
    topics = list(TOPIC_WEIGHTS.keys())
    weights = list(TOPIC_WEIGHTS.values())

    return {
        "text_length": int(
            rng.integers(ranges["text_length"][0], ranges["text_length"][1] + 1)
        ),
        "num_options": 4,
        "avg_time_taken": int(
            rng.integers(ranges["avg_time_taken"][0], ranges["avg_time_taken"][1] + 1)
        ),
        "past_accuracy": round(
            float(rng.uniform(ranges["past_accuracy"][0], ranges["past_accuracy"][1])),
            2,
        ),
        "topic": rng.choice(topics, p=weights),
        "difficulty": difficulty,
    }


def generate_dataset(
    total_rows: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate the full synthetic dataset.

    Parameters
    ----------
    total_rows : int
        Total number of question rows to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: question_id, topic, text_length,
        num_options, avg_time_taken, past_accuracy, difficulty.
    """
    validate_ratios()
    validate_topic_weights()

    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    qid = 1

    for difficulty, ratio in DIFFICULTY_RATIOS.items():
        count = int(round(total_rows * ratio))
        logger.info(
            "Generating %d '%s' questions (%.0f%% of %d)",
            count, difficulty, ratio * 100, total_rows,
        )
        for _ in range(count):
            row = generate_row(difficulty, rng)
            row["question_id"] = f"Q{qid:04d}"
            rows.append(row)
            qid += 1

    # Ensure exact total (rounding may add/remove 1)
    while len(rows) < total_rows:
        row = generate_row("Medium", rng)
        row["question_id"] = f"Q{qid:04d}"
        rows.append(row)
        qid += 1

    df = pd.DataFrame(rows)[
        [
            "question_id",
            "topic",
            "text_length",
            "num_options",
            "avg_time_taken",
            "past_accuracy",
            "difficulty",
        ]
    ]

    # Shuffle to avoid ordered blocks
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    return df


def print_summary(df: pd.DataFrame) -> None:
    """Print a statistical summary of the generated dataset."""
    logger.info("─" * 55)
    logger.info("Dataset Summary")
    logger.info("─" * 55)

    # Difficulty distribution
    dist = df["difficulty"].value_counts()
    for level in ["Easy", "Medium", "Hard"]:
        count = dist.get(level, 0)
        pct = count / len(df) * 100
        logger.info("  %-8s : %4d rows (%5.1f%%)", level, count, pct)

    # Topic distribution
    logger.info("")
    logger.info("Topic distribution:")
    for topic, count in df["topic"].value_counts().items():
        logger.info("  %-20s : %4d rows", topic, count)

    # Feature stats
    logger.info("")
    logger.info("Feature statistics:")
    for col in ["text_length", "avg_time_taken", "past_accuracy"]:
        logger.info(
            "  %-16s : mean=%.2f  std=%.2f  min=%.2f  max=%.2f",
            col, df[col].mean(), df[col].std(), df[col].min(), df[col].max(),
        )
    logger.info("─" * 55)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic DGCA question dataset for ML training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--rows", type=int, default=500,
        help="Total number of question rows to generate",
    )
    parser.add_argument(
        "--output", type=str, default="dataset.csv",
        help="Output CSV filename",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()

    if args.rows < 10:
        logger.error("--rows must be at least 10, got %d", args.rows)
        sys.exit(1)

    logger.info("Generating dataset with %d rows (seed=%d)...", args.rows, args.seed)

    df = generate_dataset(total_rows=args.rows, seed=args.seed)

    output_path = Path(args.output)
    df.to_csv(output_path, index=False)
    logger.info("✅ Saved %d rows → %s", len(df), output_path.resolve())

    print_summary(df)


if __name__ == "__main__":
    main()
