import os
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union

class SemanticExplainer:
    """
    Semantic Explainer & Nearest-Neighbor Semantic Retriever.
    Provides difficulty prediction, probability distributions, top-5 semantic neighbors,
    and embedding vector previews without hardcoded rules.
    """
    def __init__(
        self,
        embeddings_path: str = "embeddings.npy",
        dataset_path: str = "dataset_semantic.csv",
        label_encoder_path: str = "label_encoder.pkl",
        model_path: str = "hybrid_semantic_model.pkl",
        embedder_path: str = "embedder.pkl"
    ):
        print("[*] Initializing SemanticExplainer...")
        self.embeddings = np.load(embeddings_path)
        self.df = pd.read_csv(dataset_path)
        self.le = joblib.load(label_encoder_path)
        self.model = joblib.load(model_path)
        self.embedder = joblib.load(embedder_path)
        print(f"[+] Loaded artifacts: {len(self.df)} corpus questions, {self.embeddings.shape[1]}-D embeddings.")

    def explain(
        self,
        question: str,
        options: Union[List[str], Dict[str, str], None] = None,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Calculates dense embedding, hybrid features, predicted difficulty,
        class probabilities, and top-5 cosine similar training questions.
        """
        # 1. Format text
        opt_text = ""
        num_options = 4
        if isinstance(options, dict):
            num_options = len(options)
            opt_text = " ".join([f"Option {k}: {v}" for k, v in options.items()])
        elif isinstance(options, list):
            num_options = len(options)
            letters = ["A", "B", "C", "D", "E"]
            opt_text = " ".join([f"Option {letters[i]}: {opt}" for i, opt in enumerate(options) if i < len(letters)])

        semantic_input = f"{question} {opt_text}".strip()
        text_len = len(question)
        has_num = 1 if any(char.isdigit() for char in question) else 0
        
        # Structural defaults for inference
        avg_time = 45.0
        past_acc = 0.65

        # 2. Compute 384-D normalized embedding vector
        emb_vector = self.embedder.encode(
            [semantic_input],
            normalize_embeddings=True,
            convert_to_numpy=True
        )[0]

        # 3. Hybrid Feature Vector (384 + 4 = 388)
        structural_vec = np.array([text_len, num_options, avg_time, past_acc], dtype=float)
        x_hybrid = np.hstack([emb_vector, structural_vec]).reshape(1, -1)

        # 4. Predict Difficulty & Class Probabilities
        pred_label_idx = self.model.predict(x_hybrid)[0]
        pred_difficulty = self.le.inverse_transform([pred_label_idx])[0]

        probabilities = {}
        confidence = 1.0
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(x_hybrid)[0]
            for class_idx, class_name in enumerate(self.le.classes_):
                probabilities[class_name.lower()] = float(round(probs[class_idx], 4))
            confidence = float(round(float(np.max(probs)), 4))
        else:
            probabilities[pred_difficulty.lower()] = 1.0

        # 5. Top-K Cosine Similarity with Training Corpus (Dot product on normalized embeddings)
        cosine_sims = np.dot(self.embeddings, emb_vector)
        top_indices = np.argsort(cosine_sims)[::-1][:top_k]

        similar_questions = []
        for rank, idx in enumerate(top_indices, 1):
            row = self.df.iloc[idx]
            similar_questions.append({
                "rank": rank,
                "q_num": int(row.get("q_num", idx + 1)),
                "question": str(row.get("question", "")),
                "topic": str(row.get("topic", "Aviation Meteorology")),
                "difficulty": str(row.get("difficulty", "Medium")),
                "similarity_score": float(round(cosine_sims[idx], 4))
            })

        return {
            "question": question,
            "difficulty": pred_difficulty,
            "confidence": confidence,
            "probabilities": probabilities,
            "similar_questions": similar_questions,
            "embedding_preview": [float(round(v, 4)) for v in emb_vector[:20].tolist()],
            "embedding_dim": int(len(emb_vector))
        }

if __name__ == "__main__":
    explainer = SemanticExplainer()

    # 3 Questions (including 2 rephrased versions to prove SEMANTIC invariance)
    q1 = "Lowest layer of atmosphere is"
    q2_rephrased = "Which atmospheric region is situated closest to the surface of the earth?"
    q3_distinct = "What is the maximum wind speed inside a severe cyclonic storm?"

    opts_1 = {"a": "Troposphere", "b": "Tropopause", "c": "Stratosphere"}
    opts_2 = {"a": "Stratosphere", "b": "Troposphere", "c": "Mesosphere"}
    opts_3 = {"a": "17 kt", "b": "34 kt", "c": "48 kt"}

    print("\n" + "=" * 80)
    print("                SEMANTIC INVARIANCE & EXPLANATION PROOF                 ")
    print("=" * 80)

    for i, (q, opts) in enumerate([(q1, opts_1), (q2_rephrased, opts_2), (q3_distinct, opts_3)], 1):
        print(f"\n--- TEST QUESTION #{i} ---")
        print(f"Query: \"{q}\"")
        res = explainer.explain(q, opts)
        print(f"Predicted Difficulty : {res['difficulty']} (Confidence: {res['confidence']*100:.1f}%)")
        print(f"Probabilities        : {res['probabilities']}")
        print(f"First 20 Embed Dims  : {res['embedding_preview']}")
        print("Top Similar Training Questions:")
        for sim in res["similar_questions"][:3]:
            print(f"  * [Sim: {sim['similarity_score']:.4f} | Diff: {sim['difficulty']:<6}] Q: {sim['question'][:60]}...")
