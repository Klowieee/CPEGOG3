"""
Domain guard: decides whether a user question is "in scope" (about the
student handbook) BEFORE it's sent to the generator. This is what stops a
fine-tuned small model from confidently hallucinating an answer to
unrelated questions (e.g. "write me a poem about cats").

Approach: TF-IDF cosine similarity against your known FAQ questions/topics.
This needs no extra model download (scikit-learn only) and is fast enough
to run on every query. It's intentionally simple — good enough for a
class project; a production system would likely use sentence-embeddings.
"""
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

FALLBACK_MESSAGE = (
    "I can only help with questions about the student handbook "
    "(enrollment, grading, attendance, discipline, scholarships, and "
    "student organizations). Could you rephrase your question around one "
    "of those topics?"
)

DEFAULT_THRESHOLD = 0.12  # tune this after testing on real queries


class DomainGuard:
    def __init__(self, faq_path: str, threshold: float = DEFAULT_THRESHOLD):
        self.threshold = threshold
        faq_items = json.loads(Path(faq_path).read_text(encoding="utf-8"))
        self.reference_texts = [
            f"{item.get('category', '')} {item['question']}" for item in faq_items
        ]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.reference_matrix = self.vectorizer.fit_transform(self.reference_texts)

    def is_in_domain(self, query: str) -> bool:
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.reference_matrix)
        return sims.max() >= self.threshold

    def check(self, query: str):
        """Returns (in_domain: bool, fallback_message_or_none: str|None)"""
        if self.is_in_domain(query):
            return True, None
        return False, FALLBACK_MESSAGE


if __name__ == "__main__":
    # quick manual test
    guard = DomainGuard(faq_path="../data/handbook_faq_real.json")
    test_queries = [
        "How many absences before I get dropped from class?",
        "What's the weather like today?",
        "Can you write me a poem about cats?",
        "How do I renew my scholarship?",
    ]
    for q in test_queries:
        in_domain, msg = guard.check(q)
        print(f"[{'IN-DOMAIN' if in_domain else 'OFF-TOPIC '}] {q}")
        if msg:
            print(f"    -> fallback: {msg}")
