"""
retrain_pipeline.py — Periodic Retraining Pipeline for AeroBeacon
==================================================================

Pulls fresh student attempt data from MongoDB, merges it with the
existing dataset, retrains the model, and promotes or rolls back
based on accuracy comparison.

Designed to run as a nightly cron job:
    0 2 * * * cd /path/to/ml && python retrain_pipeline.py

Features:
    - Safe model rollback if accuracy degrades
    - Console alerts (Slack/email hooks stubbed for production)
    - Detailed retraining logs with timestamps
    - Scheduling metadata output
    - Configurable MongoDB URI via environment variable

Usage:
    python retrain_pipeline.py                              # default
    python retrain_pipeline.py --mongo-uri mongodb://...    # custom URI
    python retrain_pipeline.py --dataset dataset.csv        # custom CSV
    python retrain_pipeline.py --dry-run                    # simulate only
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("retrain_pipeline")

# ── Constants ────────────────────────────────────────────────────────────────
FEATURE_COLUMNS = ["text_length", "num_options", "avg_time_taken", "past_accuracy"]
TEST_SIZE = 0.2
RANDOM_STATE = 42
MIN_ACCURACY_THRESHOLD = 0.70  # Reject any model below this absolute threshold


# ── Alert System (Stub) ─────────────────────────────────────────────────────

def send_alert(
    title: str,
    message: str,
    level: str = "info",
    channel: str = "console",
) -> None:
    """
    Send an alert notification.

    In production, this would integrate with Slack / Email / PagerDuty.
    Currently logs to console with visual formatting.

    Parameters
    ----------
    title : str
        Alert title.
    message : str
        Alert body text.
    level : str
        One of 'info', 'warning', 'error', 'critical'.
    channel : str
        Target channel (currently only 'console' is implemented).
    """
    icons = {
        "info": "ℹ️",
        "warning": "⚠️",
        "error": "❌",
        "critical": "🚨",
    }
    icon = icons.get(level, "📢")

    border = "═" * 60
    logger.log(
        getattr(logging, level.upper(), logging.INFO),
        "\n%s\n%s  ALERT: %s\n%s\n%s\n%s",
        border, icon, title, border, message, border,
    )

    # ── Slack integration stub ──
    # slack_webhook = os.getenv("SLACK_WEBHOOK_URL")
    # if slack_webhook:
    #     import requests
    #     requests.post(slack_webhook, json={
    #         "text": f"{icon} *{title}*\n{message}"
    #     })

    # ── Email integration stub ──
    # smtp_host = os.getenv("SMTP_HOST")
    # if smtp_host:
    #     import smtplib
    #     from email.mime.text import MIMEText
    #     msg = MIMEText(message)
    #     msg["Subject"] = f"[AeroBeacon] {title}"
    #     msg["From"] = os.getenv("ALERT_FROM", "noreply@aerobeacon.local")
    #     msg["To"] = os.getenv("ALERT_TO", "admin@aerobeacon.local")
    #     with smtplib.SMTP(smtp_host) as server:
    #         server.send_message(msg)


# ── MongoDB Data Pull ────────────────────────────────────────────────────────

def pull_attempt_data(mongo_uri: str) -> list[dict]:
    """
    Aggregate student attempt data from MongoDB.

    Groups by questionId and computes average accuracy and time.

    Parameters
    ----------
    mongo_uri : str
        MongoDB connection URI.

    Returns
    -------
    list[dict]
        List of aggregated attempt records.
    """
    try:
        from pymongo import MongoClient
    except ImportError:
        logger.warning(
            "pymongo not installed. Skipping MongoDB pull. "
            "Install with: pip install pymongo"
        )
        return []

    if not mongo_uri:
        logger.warning("MONGO_URI not set. Skipping MongoDB data pull.")
        return []

    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        # Test connection
        client.admin.command("ping")
        db = client["aerobeacon"]

        pipeline = [
            {
                "$group": {
                    "_id": "$questionId",
                    "past_accuracy": {
                        "$avg": {"$cond": ["$isCorrect", 1, 0]}
                    },
                    "avg_time_taken": {"$avg": "$timeTakenSeconds"},
                    "attempt_count": {"$sum": 1},
                }
            },
            # Only use questions with enough attempts for statistical significance
            {"$match": {"attempt_count": {"$gte": 3}}},
        ]

        attempts = list(db.attempts.aggregate(pipeline))
        logger.info("📊 Pulled %d aggregated question records from MongoDB", len(attempts))
        client.close()
        return attempts

    except Exception as e:
        logger.error("MongoDB connection failed: %s", e)
        send_alert(
            "MongoDB Connection Failed",
            f"Could not pull attempt data: {e}\n"
            "Retraining will proceed with existing dataset only.",
            level="warning",
        )
        return []


# ── Dataset Merging ──────────────────────────────────────────────────────────

def merge_datasets(
    synthetic_path: str,
    attempts: list[dict],
    output_path: str,
) -> pd.DataFrame:
    """
    Merge real student attempt data into the synthetic dataset.

    Real data overwrites synthetic past_accuracy and avg_time_taken
    for matching question IDs.

    Parameters
    ----------
    synthetic_path : str
        Path to the original synthetic CSV.
    attempts : list[dict]
        Aggregated MongoDB attempt data.
    output_path : str
        Where to save the merged dataset.

    Returns
    -------
    pd.DataFrame
        The merged dataset.
    """
    df = pd.read_csv(synthetic_path)
    logger.info("Loaded base dataset: %d rows from %s", len(df), synthetic_path)

    updates_applied = 0
    for attempt in attempts:
        qid = str(attempt["_id"])
        mask = df["question_id"] == qid
        if mask.any():
            df.loc[mask, "past_accuracy"] = round(attempt["past_accuracy"], 2)
            df.loc[mask, "avg_time_taken"] = int(attempt["avg_time_taken"])
            updates_applied += 1

    logger.info(
        "✏️  Applied %d real-data updates (%d attempt records, %d matched)",
        updates_applied, len(attempts), updates_applied,
    )

    df.to_csv(output_path, index=False)
    logger.info("💾 Saved merged dataset → %s", output_path)
    return df


# ── Model Comparison ─────────────────────────────────────────────────────────

def compare_models(
    old_model_path: str,
    new_model_path: str,
    le_path: str,
    df: pd.DataFrame,
) -> tuple[float, float]:
    """
    Compare old vs. new model accuracy on a held-out test set.

    Parameters
    ----------
    old_model_path : str
        Path to the backed-up old model.
    new_model_path : str
        Path to the newly trained model.
    le_path : str
        Path to the label encoder.
    df : pd.DataFrame
        The full dataset (used to create a consistent test split).

    Returns
    -------
    tuple[float, float]
        (old_accuracy, new_accuracy)
    """
    old_model = joblib.load(old_model_path)
    new_model = joblib.load(new_model_path)
    le = joblib.load(le_path)

    X = df[FEATURE_COLUMNS]
    y = le.transform(df["difficulty"])

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )

    old_acc = accuracy_score(y_test, old_model.predict(X_test))
    new_acc = accuracy_score(y_test, new_model.predict(X_test))

    return old_acc, new_acc


# ── Retrain Hook ─────────────────────────────────────────────────────────────

def notify_ml_service(service_url: str = "http://localhost:5001") -> None:
    """
    Tell the running Flask ML service to reload its model from disk.

    Parameters
    ----------
    service_url : str
        Base URL of the ML Flask service.
    """
    try:
        import requests
        resp = requests.post(f"{service_url}/retrain-hook", timeout=10)
        if resp.status_code == 200:
            logger.info("🔄 ML service reloaded model successfully")
        else:
            logger.warning("ML service reload returned status %d", resp.status_code)
    except ImportError:
        logger.info("'requests' not installed — skipping service reload notification")
    except Exception as e:
        logger.warning("Could not notify ML service: %s", e)


# ── Scheduling Metadata ─────────────────────────────────────────────────────

def save_schedule_metadata(output_dir: Path, result: dict) -> None:
    """
    Write a JSON file with the retraining result for monitoring dashboards.

    Parameters
    ----------
    output_dir : Path
        Directory to write retrain_log.json.
    result : dict
        Retraining result data.
    """
    log_path = output_dir / "retrain_log.json"

    # Load existing log entries
    entries = []
    if log_path.exists():
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                entries = json.load(f)
        except (json.JSONDecodeError, IOError):
            entries = []

    entries.append(result)

    # Keep last 100 entries
    entries = entries[-100:]

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)

    logger.info("📋 Saved retrain log → %s", log_path)


# ── Main Pipeline ────────────────────────────────────────────────────────────

def run_pipeline(args: argparse.Namespace) -> None:
    """
    Execute the full retrain pipeline.

    Steps:
        1. Pull new attempt data from MongoDB (if available)
        2. Merge with existing dataset
        3. Backup current model
        4. Retrain model
        5. Compare old vs new accuracy
        6. Promote or rollback
        7. Notify ML service to reload
        8. Log results
    """
    started_at = datetime.now(timezone.utc)
    model_dir = Path(args.model_dir)
    dataset_path = model_dir / args.dataset
    merged_path = model_dir / "dataset_real.csv"
    model_path = model_dir / "difficulty_model.pkl"
    backup_path = model_dir / "difficulty_model_old.pkl"
    le_path = model_dir / "label_encoder.pkl"

    logger.info("═" * 60)
    logger.info("🔁  AeroBeacon Retrain Pipeline — Starting")
    logger.info("═" * 60)

    result = {
        "started_at": started_at.isoformat(),
        "status": "unknown",
        "old_accuracy": None,
        "new_accuracy": None,
        "action": None,
        "error": None,
    }

    try:
        # 1. Pull MongoDB data
        mongo_uri = args.mongo_uri or os.getenv("MONGO_URI", "")
        attempts = pull_attempt_data(mongo_uri)

        # 2. Merge datasets
        if not dataset_path.exists():
            raise FileNotFoundError(f"Base dataset not found: {dataset_path}")

        df = merge_datasets(str(dataset_path), attempts, str(merged_path))

        # 3. Backup current model
        if model_path.exists():
            shutil.copy2(str(model_path), str(backup_path))
            logger.info("💾 Backed up current model → %s", backup_path)
        else:
            logger.warning("No existing model to backup. This appears to be a first run.")

        if args.dry_run:
            logger.info("🏁 Dry run — skipping actual training.")
            result["status"] = "dry_run"
            result["action"] = "none"
            save_schedule_metadata(model_dir, result)
            return

        # 4. Retrain
        logger.info("🏋️  Retraining model...")
        train_cmd = [
            sys.executable, str(model_dir / "train_model.py"),
            "--dataset", str(merged_path),
            "--output-dir", str(model_dir),
        ]
        proc = subprocess.run(
            train_cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"train_model.py failed (exit {proc.returncode}):\n"
                f"STDOUT: {proc.stdout[-500:]}\n"
                f"STDERR: {proc.stderr[-500:]}"
            )
        logger.info("Training output:\n%s", proc.stdout[-1000:])

        # 5. Compare models
        if backup_path.exists():
            old_acc, new_acc = compare_models(
                str(backup_path), str(model_path), str(le_path), df,
            )
            result["old_accuracy"] = round(old_acc, 4)
            result["new_accuracy"] = round(new_acc, 4)

            logger.info("📊 Old accuracy: %.4f | New accuracy: %.4f", old_acc, new_acc)

            # 6. Decision: promote or rollback
            if new_acc < old_acc:
                shutil.copy2(str(backup_path), str(model_path))
                result["status"] = "rolled_back"
                result["action"] = "rollback"
                send_alert(
                    "Model Rollback — Accuracy Regression",
                    f"New model ({new_acc:.4f}) performed worse than old ({old_acc:.4f}).\n"
                    f"Rolled back to previous model.\n"
                    f"Timestamp: {started_at.isoformat()}",
                    level="warning",
                )
            elif new_acc < MIN_ACCURACY_THRESHOLD:
                shutil.copy2(str(backup_path), str(model_path))
                result["status"] = "rolled_back"
                result["action"] = "rollback_below_threshold"
                send_alert(
                    "Model Rollback — Below Minimum Threshold",
                    f"New model accuracy ({new_acc:.4f}) is below minimum "
                    f"threshold ({MIN_ACCURACY_THRESHOLD:.4f}).\n"
                    f"Rolled back to previous model ({old_acc:.4f}).",
                    level="error",
                )
            else:
                result["status"] = "promoted"
                result["action"] = "promote"
                improvement = new_acc - old_acc
                send_alert(
                    "Model Promoted Successfully",
                    f"New model ({new_acc:.4f}) promoted over old ({old_acc:.4f}).\n"
                    f"Improvement: +{improvement:.4f}",
                    level="info",
                )
        else:
            result["status"] = "first_train"
            result["action"] = "promote"
            logger.info("First training run — no comparison needed.")

        # 7. Notify ML service to reload
        notify_ml_service()

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        logger.exception("Pipeline failed")
        send_alert(
            "Retrain Pipeline Failed",
            f"Error: {e}\nTimestamp: {started_at.isoformat()}",
            level="critical",
        )

    finally:
        result["finished_at"] = datetime.now(timezone.utc).isoformat()
        duration = (
            datetime.fromisoformat(result["finished_at"])
            - datetime.fromisoformat(result["started_at"])
        ).total_seconds()
        result["duration_seconds"] = round(duration, 1)

        save_schedule_metadata(model_dir, result)

        logger.info("═" * 60)
        logger.info(
            "🏁  Pipeline finished: %s (%.1fs)",
            result["status"], duration,
        )
        logger.info("═" * 60)


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AeroBeacon model retraining pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mongo-uri", type=str, default="",
        help="MongoDB connection URI (overrides MONGO_URI env var)",
    )
    parser.add_argument(
        "--dataset", type=str, default="dataset.csv",
        help="Base dataset filename (relative to model-dir)",
    )
    parser.add_argument(
        "--model-dir", type=str, default=".",
        help="Directory containing model files",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Simulate pipeline without training",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
