import json
import random
from typing import List, Dict, Any, Optional
from semantic_embedder import SemanticEmbedder

class QuizGenerator:
    """
    Semantic & Chapter-based Quiz Generator for Aviation Meteorology.
    Generates practice exams, topic-wise tests, and concept-targeted adaptive quizzes.
    """
    def __init__(self, dataset_path: str = "Meteorology formate ml model train.json"):
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        self.metadata = data.get("metadata", {})
        self.questions: List[Dict[str, Any]] = data.get("questions", [])
        self.embedder: Optional[SemanticEmbedder] = None

        # Build chapter index
        self.chapters: Dict[str, List[Dict[str, Any]]] = {}
        for q in self.questions:
            chap = q.get("chapter", "General Meteorology")
            if chap not in self.chapters:
                self.chapters[chap] = []
            self.chapters[chap].append(q)

    def get_available_chapters(self) -> List[str]:
        return sorted(list(self.chapters.keys()))

    def generate_random_quiz(self, num_questions: int = 10, chapter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Generates a standard multiple-choice quiz.
        """
        pool = self.questions
        if chapter and chapter in self.chapters:
            pool = self.chapters[chapter]

        count = min(num_questions, len(pool))
        selected = random.sample(pool, count)
        
        quiz = []
        for i, q in enumerate(selected, 1):
            quiz.append({
                "quiz_id": i,
                "original_id": q.get("id"),
                "chapter": q.get("chapter"),
                "question": q.get("question"),
                "options": q.get("options", {}),
                "correct_answer": q.get("answer", "").strip().lower(),
                "explanation": q.get("explanation", "")
            })
        return quiz

    def generate_semantic_quiz(self, concept_query: str, num_questions: int = 5) -> List[Dict[str, Any]]:
        """
        Generates a targeted quiz semantically focused around a specific topic
        (e.g., 'altimeter errors in cold weather' or 'microburst and squalls').
        """
        if self.embedder is None:
            self.embedder = SemanticEmbedder()
            self.embedder.build_index("Meteorology formate ml model train.json")

        hits = self.embedder.search_semantic(concept_query, top_k=num_questions * 2)
        selected_hits = hits[:num_questions]

        quiz = []
        for i, hit in enumerate(selected_hits, 1):
            quiz.append({
                "quiz_id": i,
                "original_id": hit.get("id"),
                "chapter": hit.get("chapter"),
                "question": hit.get("question"),
                "options": hit.get("options", {}),
                "correct_answer": hit.get("answer_key", "").strip().lower(),
                "explanation": hit.get("explanation", ""),
                "relevance_score": hit.get("cosine_similarity", 0.0)
            })
        return quiz

if __name__ == "__main__":
    qgen = QuizGenerator()
    print("Available Chapters:")
    for chap in qgen.get_available_chapters():
        print(f" - {chap} ({len(qgen.chapters[chap])} questions)")

    print("\n--- Generating 3 Questions on 'Atmospheric Pressure' ---")
    sample_quiz = qgen.generate_random_quiz(num_questions=3, chapter="Atmospheric Pressure")
    for q in sample_quiz:
        print(f"\nQ{q['quiz_id']}: {q['question']}")
        for opt_key, opt_val in q['options'].items():
            print(f"   ({opt_key}) {opt_val}")
        print(f"Correct: ({q['correct_answer']}) | Explanation: {q['explanation']}")
