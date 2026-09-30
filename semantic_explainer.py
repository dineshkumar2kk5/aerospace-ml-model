import os
import torch
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import CrossEncoder
from semantic_embedder import SemanticEmbedder

class SemanticExplainer:
    """
    Neural Semantic Reasoner & Cross-Encoder Reranker.
    Uses deep transformer cross-attention to score query-document semantic relevance
    and synthesizes grounded aviation meteorological explanations.
    """
    def __init__(
        self,
        embedder: Optional[SemanticEmbedder] = None,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if embedder is None:
            print("[*] Initializing underlying Semantic Embedder...")
            self.embedder = SemanticEmbedder(device=self.device)
            self.embedder.build_index("Meteorology formate ml model train.json")
        else:
            self.embedder = embedder

        print(f"[*] Loading Neural Cross-Encoder Reranker: {reranker_model} on {self.device.upper()}")
        self.reranker = CrossEncoder(reranker_model, device=self.device)

    def explain(
        self,
        user_query: str,
        retrieval_k: int = 10,
        rerank_top_k: int = 3
    ) -> Dict[str, Any]:
        """
        End-to-end Semantic ML pipeline:
        1. Bi-Encoder Semantic Vector Search (Stage 1 candidate generation)
        2. Cross-Encoder Deep Neural Scoring (Stage 2 reranking)
        3. Semantic Explanation Synthesis (Stage 3)
        """
        user_query_clean = user_query.strip()
        if not user_query_clean:
            return {"status": "error", "message": "Query cannot be empty."}

        # Stage 1: Vector Search in Latent Embedding Space
        candidates = self.embedder.search_semantic(user_query_clean, top_k=retrieval_k)
        if not candidates:
            return {
                "status": "not_found",
                "query": user_query_clean,
                "message": "No relevant meteorological concepts found."
            }

        # Stage 2: Cross-Encoder Transformer Cross-Attention
        cross_pairs = []
        for cand in candidates:
            # Pair query with full semantic textbook context
            cross_pairs.append([user_query_clean, cand["semantic_context"]])

        neural_scores = self.reranker.predict(cross_pairs)

        # Rank candidates by neural cross-attention logits
        sorted_indices = np.argsort(neural_scores)[::-1][:rerank_top_k]

        ranked_results = []
        for rank, s_idx in enumerate(sorted_indices, 1):
            cand = dict(candidates[s_idx])
            cand["rank"] = rank
            cand["neural_score"] = float(round(neural_scores[s_idx], 4))
            # Convert logit to normalized sigmoid confidence
            prob = 1.0 / (1.0 + np.exp(-float(neural_scores[s_idx])))
            cand["confidence_prob"] = float(round(prob, 4))
            ranked_results.append(cand)

        top_match = ranked_results[0]

        return {
            "status": "success",
            "query": user_query_clean,
            "top_concept": {
                "id": top_match["id"],
                "chapter": top_match["chapter"],
                "question": top_match["question"],
                "options": top_match["options"],
                "correct_option": top_match["answer_key"],
                "correct_answer": top_match["answer_text"],
                "scientific_explanation": top_match["explanation"],
                "cosine_similarity": top_match["cosine_similarity"],
                "neural_relevance_score": top_match["neural_score"],
                "confidence_level": f"{top_match['confidence_prob'] * 100:.1f}%"
            },
            "alternative_matches": ranked_results[1:],
            "semantic_summary": (
                f"Based on Aviation Meteorology ({top_match['chapter']}): "
                f"For '{top_match['question']}', the correct concept is "
                f"'{top_match['answer_text']}'. Reason: {top_match['explanation']}"
            )
        }

if __name__ == "__main__":
    explainer = SemanticExplainer()
    
    test_queries = [
        "Why is troposphere height more near equator?",
        "what happens when an airplane flies from high pressure region to low pressure without resetting altimeter?",
        "which ice accretion is most dangerous to aircraft controls?"
    ]

    for q in test_queries:
        print("\n" + "="*80)
        print(f"QUERY: {q}")
        print("="*80)
        result = explainer.explain(q)
        top = result["top_concept"]
        print(f"Matched Chapter : {top['chapter']}")
        print(f"Matched Q&A     : {top['question']} -> ({top['correct_option']}) {top['correct_answer']}")
        print(f"Explanation     : {top['scientific_explanation']}")
        print(f"Neural Score    : {top['neural_relevance_score']} (Confidence: {top['confidence_level']})")
