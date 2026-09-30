import os
import time
import logging
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from semantic_explainer import SemanticExplainer
from hybrid_train import train_hybrid_models

# Configure structured request logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("SemanticMLService")

app = Flask(__name__)
CORS(app)

# Global Explainer instance
explainer = None

def init_service():
    global explainer
    logger.info("[*] Initializing SemanticExplainer backend...")
    explainer = SemanticExplainer()
    logger.info("[+] Semantic ML Service initialized successfully!")

# Initialize upon startup
try:
    init_service()
except Exception as e:
    logger.warning(f"[!] Initial load failed: {e}. Retrain or check artifacts.")

@app.before_request
def log_request():
    request._start_time = time.time()

@app.after_request
def log_response(response):
    duration = (time.time() - getattr(request, "_start_time", time.time())) * 1000
    logger.info(f"{request.method} {request.path} - Status: {response.status_code} - {duration:.2f}ms")
    return response

@app.route("/health", methods=["GET"])
def health():
    if explainer is None:
        return jsonify({"status": "degraded", "message": "Model not loaded"}), 503
    return jsonify({
        "status": "healthy",
        "service": "Semantic ML Difficulty & Explanation API",
        "embedding_dim": explainer.embeddings.shape[1],
        "training_size": len(explainer.df),
        "target_classes": explainer.le.classes_.tolist()
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        if not data or "question" not in data:
            return jsonify({"error": "Missing 'question' in request body"}), 400

        question = str(data["question"]).strip()
        options = data.get("options", None)
        top_k = int(data.get("top_k", 5))

        res = explainer.explain(question=question, options=options, top_k=top_k)
        return jsonify({
            "status": "success",
            "question": res["question"],
            "difficulty": res["difficulty"],
            "confidence": res["confidence"],
            "probabilities": res["probabilities"],
            "similar_questions": res["similar_questions"]
        })
    except Exception as e:
        logger.error(f"Error in /predict: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route("/predict-batch", methods=["POST"])
def predict_batch():
    """
    Fully Vectorized Batch Inference (Zero Python loops for predictions)
    """
    try:
        data = request.get_json(force=True)
        items = data.get("items", [])
        if not items:
            return jsonify({"error": "Missing or empty 'items' list"}), 400

        questions = []
        semantic_texts = []
        structural_rows = []

        for item in items:
            q_text = str(item.get("question", "")).strip()
            options = item.get("options", {})
            opt_str = ""
            num_options = 4
            if isinstance(options, dict):
                num_options = len(options)
                opt_str = " ".join([f"Option {k}: {v}" for k, v in options.items()])
            elif isinstance(options, list):
                num_options = len(options)
                letters = ["A", "B", "C", "D", "E"]
                opt_str = " ".join([f"Option {letters[i]}: {opt}" for i, opt in enumerate(options) if i < len(letters)])

            text_len = len(q_text)
            avg_time = float(item.get("avg_time_taken", 45.0))
            past_acc = float(item.get("past_accuracy", 0.65))

            questions.append(q_text)
            semantic_texts.append(f"{q_text} {opt_str}".strip())
            structural_rows.append([text_len, num_options, avg_time, past_acc])

        # 1. Vectorized Batch Encoding
        batch_embeddings = explainer.embedder.encode(
            semantic_texts,
            batch_size=64,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        # 2. Vectorized Hybrid Feature Matrix
        batch_structural = np.array(structural_rows, dtype=float)
        X_batch_hybrid = np.hstack([batch_embeddings, batch_structural])

        # 3. Vectorized Prediction
        pred_label_indices = explainer.model.predict(X_batch_hybrid)
        pred_difficulties = explainer.le.inverse_transform(pred_label_indices)

        # 4. Vectorized Class Probabilities
        if hasattr(explainer.model, "predict_proba"):
            batch_probs = explainer.model.predict_proba(X_batch_hybrid)
            confidences = np.max(batch_probs, axis=1)
        else:
            batch_probs = None
            confidences = np.ones(len(items))

        # 5. Build results
        results = []
        for i in range(len(items)):
            probs_dict = {}
            if batch_probs is not None:
                for c_idx, c_name in enumerate(explainer.le.classes_):
                    probs_dict[c_name.lower()] = float(round(batch_probs[i][c_idx], 4))

            results.append({
                "index": i,
                "question": questions[i],
                "difficulty": str(pred_difficulties[i]),
                "confidence": float(round(confidences[i], 4)),
                "probabilities": probs_dict
            })

        return jsonify({
            "status": "success",
            "count": len(results),
            "predictions": results
        })

    except Exception as e:
        logger.error(f"Error in /predict-batch: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route("/explain", methods=["POST"])
def explain_endpoint():
    try:
        data = request.get_json(force=True)
        question = str(data.get("question", "")).strip()
        options = data.get("options", None)
        top_k = int(data.get("top_k", 10))

        res = explainer.explain(question=question, options=options, top_k=top_k)
        return jsonify({
            "status": "success",
            "question": res["question"],
            "difficulty": res["difficulty"],
            "confidence": res["confidence"],
            "probabilities": res["probabilities"],
            "embedding_preview_20d": res["embedding_preview"],
            "embedding_dim": res["embedding_dim"],
            "top_similar_questions": res["similar_questions"]
        })
    except Exception as e:
        logger.error(f"Error in /explain: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route("/retrain-hook", methods=["POST"])
def retrain_hook():
    try:
        logger.info("[*] Hot-reload retraining triggered via /retrain-hook...")
        train_hybrid_models()
        init_service()
        return jsonify({
            "status": "success",
            "message": "Hybrid model retrained and hot-reloaded successfully!",
            "training_size": len(explainer.df)
        })
    except Exception as e:
        logger.error(f"Error in /retrain-hook: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    logger.info("[*] Starting Flask Semantic ML Service on port 5002...")
    app.run(host="0.0.0.0", port=5002, debug=False)
