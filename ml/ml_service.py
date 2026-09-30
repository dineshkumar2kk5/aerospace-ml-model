"""
ml_service.py — Flask Microservice for AeroBeacon Difficulty Prediction
========================================================================

Serves the trained RandomForest model via REST endpoints. Designed for
local, offline deployment as part of the AeroBeacon exam platform.

Endpoints:
    GET  /health         → Service health + model metadata
    POST /predict        → Predict difficulty for a single question
    POST /predict-batch  → Predict difficulty for multiple questions
    GET  /model-info     → Detailed model metadata
    POST /retrain-hook   → Trigger model reload from disk (after retrain)

Features:
    - Request/response logging middleware
    - Input validation with descriptive errors
    - Simple in-memory rate limiting (token bucket)
    - Model metadata served alongside predictions
    - Graceful error handling

Usage:
    python ml_service.py                        # default port 5001
    python ml_service.py --port 5050            # custom port
    python ml_service.py --model-dir ./models   # custom model path
"""

import argparse
import json
import logging
import os
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, Response, g, jsonify, request

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ml_service")

# ── App Setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)

# Global model state
_model = None
_label_encoder = None
_metadata = {}
_model_dir = Path(".")
_model_load_lock = threading.Lock()

# ── Rate Limiter ─────────────────────────────────────────────────────────────

class TokenBucketRateLimiter:
    """
    Simple in-memory token-bucket rate limiter.

    Each IP address gets its own bucket. Tokens refill at a constant rate.
    Thread-safe via a reentrant lock.

    Parameters
    ----------
    max_tokens : int
        Maximum burst capacity per IP.
    refill_rate : float
        Tokens added per second per IP.
    """

    def __init__(self, max_tokens: int = 30, refill_rate: float = 2.0):
        self.max_tokens = max_tokens
        self.refill_rate = refill_rate
        self._buckets: dict[str, dict] = defaultdict(
            lambda: {"tokens": max_tokens, "last_refill": time.time()}
        )
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        """
        Check if a request from `key` is allowed.

        Returns True and consumes a token if allowed, False otherwise.
        """
        with self._lock:
            bucket = self._buckets[key]
            now = time.time()
            elapsed = now - bucket["last_refill"]
            bucket["tokens"] = min(
                self.max_tokens,
                bucket["tokens"] + elapsed * self.refill_rate,
            )
            bucket["last_refill"] = now

            if bucket["tokens"] >= 1.0:
                bucket["tokens"] -= 1.0
                return True
            return False


rate_limiter = TokenBucketRateLimiter(max_tokens=30, refill_rate=2.0)

# ── Model Loading ───────────────────────────────────────────────────────────

def load_model(model_dir: Path) -> None:
    """
    Load (or reload) the model, encoder, and metadata from disk.

    Thread-safe: uses a lock so hot-reloads during retraining are safe.

    Parameters
    ----------
    model_dir : Path
        Directory containing difficulty_model.pkl, label_encoder.pkl,
        and model_metadata.json.

    Raises
    ------
    FileNotFoundError
        If required model files are missing.
    """
    global _model, _label_encoder, _metadata, _model_dir

    model_path = model_dir / "difficulty_model.pkl"
    le_path = model_dir / "label_encoder.pkl"
    meta_path = model_dir / "model_metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not le_path.exists():
        raise FileNotFoundError(f"Label encoder not found: {le_path}")

    with _model_load_lock:
        _model = joblib.load(model_path)
        _label_encoder = joblib.load(le_path)
        _model_dir = model_dir

        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                _metadata = json.load(f)
        else:
            _metadata = {"note": "metadata file not found"}

        logger.info(
            "✅ Loaded model: %s (accuracy: %s, trained: %s)",
            _metadata.get("model_type", "unknown"),
            _metadata.get("accuracy", "?"),
            _metadata.get("trained_at", "?"),
        )


# ── Input Validation ────────────────────────────────────────────────────────

REQUIRED_FIELDS = {"text_length", "num_options", "avg_time_taken", "past_accuracy"}
FIELD_RANGES = {
    "text_length": (1, 2000),
    "num_options": (2, 10),
    "avg_time_taken": (1, 600),
    "past_accuracy": (0.0, 1.0),
}


def validate_question_input(data: dict) -> tuple[bool, str]:
    """
    Validate a single question input dictionary.

    Parameters
    ----------
    data : dict
        Input features for a single question.

    Returns
    -------
    tuple[bool, str]
        (is_valid, error_message). error_message is empty if valid.
    """
    if not isinstance(data, dict):
        return False, "Input must be a JSON object"

    missing = REQUIRED_FIELDS - set(data.keys())
    if missing:
        return False, f"Missing required fields: {sorted(missing)}"

    for field, (lo, hi) in FIELD_RANGES.items():
        val = data[field]
        if not isinstance(val, (int, float)):
            return False, f"'{field}' must be a number, got {type(val).__name__}"
        if val < lo or val > hi:
            return False, f"'{field}' must be between {lo} and {hi}, got {val}"

    return True, ""


# ── Prediction Logic ────────────────────────────────────────────────────────

def predict_one(data: dict) -> dict:
    """
    Predict difficulty for a single question.

    Parameters
    ----------
    data : dict
        Must contain: text_length, num_options, avg_time_taken, past_accuracy.

    Returns
    -------
    dict
        { difficulty: str, confidence: float, probabilities: dict }
    """
    features = np.array([[
        data["text_length"],
        data["num_options"],
        data["avg_time_taken"],
        data["past_accuracy"],
    ]])

    prediction = _model.predict(features)[0]
    probabilities = _model.predict_proba(features)[0]

    label = _label_encoder.inverse_transform([prediction])[0]
    confidence = float(np.max(probabilities))

    # Build probability map for all classes
    prob_map = {
        _label_encoder.inverse_transform([i])[0]: round(float(p), 4)
        for i, p in enumerate(probabilities)
    }

    return {
        "difficulty": label,
        "confidence": round(confidence, 4),
        "probabilities": prob_map,
    }


# ── Middleware ───────────────────────────────────────────────────────────────

@app.before_request
def before_request_handler():
    """Log incoming requests and apply rate limiting."""
    g.start_time = time.time()

    # Skip rate limiting for health check
    if request.path == "/health":
        return

    client_ip = request.remote_addr or "unknown"
    if not rate_limiter.allow(client_ip):
        logger.warning("🚫 Rate limited: %s %s from %s", request.method, request.path, client_ip)
        return jsonify({
            "error": "Rate limit exceeded. Please slow down.",
            "retry_after_seconds": 1,
        }), 429


@app.after_request
def after_request_handler(response: Response) -> Response:
    """Log response time and status."""
    duration = (time.time() - g.start_time) * 1000  # ms
    logger.info(
        "%s %s → %d (%.1fms)",
        request.method,
        request.path,
        response.status_code,
        duration,
    )
    response.headers["X-Response-Time"] = f"{duration:.1f}ms"
    return response


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint.

    Returns service status, model metadata, and uptime info.
    """
    return jsonify({
        "status": "ok",
        "service": "aerobeacon-ml",
        "model_loaded": _model is not None,
        "model_type": _metadata.get("model_type", "unknown"),
        "model_accuracy": _metadata.get("accuracy", None),
        "trained_at": _metadata.get("trained_at", None),
        "labels": _metadata.get("labels", []),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


@app.route("/model-info", methods=["GET"])
def model_info():
    """
    Return detailed model metadata.

    Includes hyperparameters, feature names, training timestamp, etc.
    """
    if not _metadata:
        return jsonify({"error": "No model metadata available"}), 503

    return jsonify({
        "metadata": _metadata,
        "features": list(REQUIRED_FIELDS),
        "field_ranges": FIELD_RANGES,
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict difficulty for a single question.

    Request body (JSON):
        {
            "text_length": 180,
            "num_options": 4,
            "avg_time_taken": 55,
            "past_accuracy": 0.62
        }

    Response (JSON):
        {
            "difficulty": "Medium",
            "confidence": 0.87,
            "probabilities": { "Easy": 0.05, "Medium": 0.87, "Hard": 0.08 }
        }
    """
    if _model is None:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Request body must be valid JSON"}), 400

    valid, error_msg = validate_question_input(data)
    if not valid:
        return jsonify({"error": error_msg}), 422

    try:
        result = predict_one(data)
        return jsonify(result)
    except Exception as e:
        logger.exception("Prediction error")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/predict-batch", methods=["POST"])
def predict_batch():
    """
    Predict difficulty for a batch of questions.

    Request body (JSON):
        {
            "questions": [
                { "text_length": 100, "num_options": 4, "avg_time_taken": 30, "past_accuracy": 0.85 },
                { "text_length": 300, "num_options": 4, "avg_time_taken": 110, "past_accuracy": 0.2 }
            ]
        }

    Response (JSON):
        [
            { "difficulty": "Easy", "confidence": 0.92, "probabilities": {...} },
            { "difficulty": "Hard", "confidence": 0.88, "probabilities": {...} }
        ]
    """
    if _model is None:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Request body must be valid JSON"}), 400

    if "questions" not in data:
        return jsonify({"error": "Missing 'questions' array in request body"}), 422

    questions = data["questions"]
    if not isinstance(questions, list):
        return jsonify({"error": "'questions' must be an array"}), 422

    if len(questions) == 0:
        return jsonify({"error": "'questions' array must not be empty"}), 422

    if len(questions) > 5000:
        return jsonify({"error": f"Batch too large: {len(questions)} (max 5000)"}), 422

    # Validate each question
    errors = []
    for i, q in enumerate(questions):
        valid, msg = validate_question_input(q)
        if not valid:
            errors.append({"index": i, "error": msg})

    if errors:
        return jsonify({"error": "Validation failed for some questions", "details": errors}), 422

    try:
        import pandas as pd
        feature_cols = ["text_length", "num_options", "avg_time_taken", "past_accuracy"]
        df_features = pd.DataFrame(
            [[q[col] for col in feature_cols] for q in questions],
            columns=feature_cols
        )
        predictions = _model.predict(df_features)
        probabilities = _model.predict_proba(df_features)
        labels = _label_encoder.inverse_transform(predictions)
        classes = _label_encoder.classes_

        results = []
        for i in range(len(questions)):
            probs = probabilities[i]
            confidence = float(np.max(probs))
            prob_map = {classes[j]: round(float(probs[j]), 4) for j in range(len(classes))}
            results.append({
                "difficulty": labels[i],
                "confidence": round(confidence, 4),
                "probabilities": prob_map,
            })
        return jsonify(results)
    except Exception as e:
        logger.exception("Batch prediction error")
        return jsonify({"error": f"Batch prediction failed: {str(e)}"}), 500


@app.route("/retrain-hook", methods=["POST"])
def retrain_hook():
    """
    Reload model from disk after a retrain.

    This endpoint is called by retrain_pipeline.py after
    successfully updating the model files.
    """
    try:
        load_model(_model_dir)
        return jsonify({
            "status": "reloaded",
            "model_type": _metadata.get("model_type"),
            "accuracy": _metadata.get("accuracy"),
            "trained_at": _metadata.get("trained_at"),
        })
    except Exception as e:
        logger.exception("Model reload failed")
        return jsonify({"error": f"Reload failed: {str(e)}"}), 500


# ── Error Handlers ───────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed"}), 405


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AeroBeacon ML prediction microservice.",
    )
    parser.add_argument("--port", type=int, default=5001, help="Port to listen on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument(
        "--model-dir", type=str, default=".",
        help="Directory containing model .pkl files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)

    try:
        load_model(model_dir)
    except FileNotFoundError as e:
        logger.error("❌ %s", e)
        logger.error(
            "Run 'python generate_dataset.py && python train_model.py' first."
        )
        sys.exit(1)

    logger.info("🚀 Starting ML service on %s:%d", args.host, args.port)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
