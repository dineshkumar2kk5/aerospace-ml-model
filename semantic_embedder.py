import json
import os
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer

class SemanticEmbedder:
    """
    Dense Neural Vector Embedder for Aviation Meteorology.
    Uses Sentence Transformers to map concepts and questions into high-dimensional latent space.
    """
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_dir: str = "./semantic_cache",
        device: Optional[str] = None
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"[*] SemanticEmbedder initializing on device: {self.device.upper()}")
        print(f"[*] Loading Dense Bi-Encoder Model: {model_name}")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.documents: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None

    def build_index(self, json_path: str, force_rebuild: bool = False):
        """
        Builds or loads cached embeddings from disk.
        """
        emb_file = self.cache_dir / "embeddings.npy"
        meta_file = self.cache_dir / "metadata.json"

        if not force_rebuild and emb_file.exists() and meta_file.exists():
            print(f"[+] Loading cached semantic index from {self.cache_dir}...")
            self.embeddings = np.load(emb_file)
            with open(meta_file, "r", encoding="utf-8") as f:
                self.documents = json.load(f)
            print(f"[+] Cached index loaded successfully! Total records: {len(self.documents)}, Dim: {self.embeddings.shape[1]}")
            return

        print(f"[*] Ingesting knowledge base from: {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        questions = data.get("questions", [])
        corpus_texts = []
        self.documents = []

        print(f"[*] Processing {len(questions)} meteorological questions & explanations...")
        for item in questions:
            ans_key = str(item.get("answer", "")).strip().lower()
            options = item.get("options", {})
            correct_text = options.get(ans_key, "")
            chapter = item.get("chapter", "Aviation Meteorology")
            question = item.get("question", "")
            explanation = item.get("explanation", "")

            # Compose high-information density semantic context
            semantic_doc = (
                f"Chapter: {chapter}. "
                f"Question: {question} "
                f"Correct Answer ({ans_key}): {correct_text}. "
                f"Explanation: {explanation}"
            )

            corpus_texts.append(semantic_doc)
            self.documents.append({
                "id": item.get("id"),
                "chapter": chapter,
                "question": question,
                "options": options,
                "answer_key": ans_key,
                "answer_text": correct_text,
                "explanation": explanation,
                "semantic_context": semantic_doc
            })

        print(f"[*] Encoding {len(corpus_texts)} entries into dense vector space (L2-normalized)...")
        self.embeddings = self.model.encode(
            corpus_texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        # Cache to disk for instant subsequent loads
        np.save(emb_file, self.embeddings)
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(self.documents, f, indent=2)

        print(f"[+] Semantic Index created & cached! Shape: {self.embeddings.shape}")

    def search_semantic(self, query: str, top_k: int = 10, threshold: float = 0.25) -> List[Dict[str, Any]]:
        """
        Retrieves top_k closest items in latent semantic space via Cosine Similarity.
        """
        if self.embeddings is None:
            raise ValueError("Index not loaded. Run build_index first.")

        # Encode query to normalized 384-d vector
        query_vec = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        
        # Dot product of normalized vectors = Cosine Similarity
        cosine_sims = np.dot(self.embeddings, query_vec)
        
        # Sort descending
        top_indices = np.argsort(cosine_sims)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            score = float(cosine_sims[idx])
            if score >= threshold:
                doc = dict(self.documents[idx])
                doc["cosine_similarity"] = round(score, 4)
                results.append(doc)

        return results

if __name__ == "__main__":
    embedder = SemanticEmbedder()
    embedder.build_index("Meteorology formate ml model train.json")
    
    # Test sample search
    sample_query = "What happens to the tropopause boundary over the equator during summer?"
    hits = embedder.search_semantic(sample_query, top_k=3)
    print(f"\nQuery: {sample_query}")
    for i, hit in enumerate(hits, 1):
        print(f"\n[{i}] Cosine Score: {hit['cosine_similarity']}")
        print(f"    Chapter: {hit['chapter']}")
        print(f"    Question: {hit['question']}")
        print(f"    Answer: ({hit['answer_key']}) {hit['answer_text']}")
        print(f"    Explanation: {hit['explanation']}")
