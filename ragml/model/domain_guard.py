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
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

FALLBACK_MESSAGE = (
    "I can only help with questions about the student handbook "
    "(enrollment, grading, attendance, discipline, scholarships, and "
    "student organizations). Could you rephrase your question around one "
    "of those topics?"
)

DEFAULT_THRESHOLD = 0.12  # tune this after testing on real queries

_WORD_RE = re.compile(r"[a-z']+")


def _naive_stem(word: str) -> str:
    """Minimal suffix stripping so plurals match their singular form
    (e.g. 'failures' <-> 'failure', 'units' <-> 'unit'). Not a real
    stemmer — just enough to stop exact-token TF-IDF matching from
    scoring a correct FAQ at 0.0 purely because the question used a
    plural where the FAQ used a singular, or vice versa."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith(("ses", "xes", "zes", "ches", "shes")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _tokenize(text: str):
    words = _WORD_RE.findall(text.lower())
    return [_naive_stem(w) for w in words if w not in ENGLISH_STOP_WORDS]


class DomainGuard:
    def __init__(self, faq_path: str, threshold: float = DEFAULT_THRESHOLD):
        self.threshold = threshold
        self.faq_items = json.loads(Path(faq_path).read_text(encoding="utf-8"))
        self.reference_texts = [
            f"{item.get('category', '')} {item['question']}" for item in self.faq_items
        ]
        self.vectorizer = TfidfVectorizer(tokenizer=_tokenize, token_pattern=None,
                                           ngram_range=(1, 2))
        self.reference_matrix = self.vectorizer.fit_transform(self.reference_texts)

    def _similarities(self, query: str):
        query_vec = self.vectorizer.transform([query])
        return cosine_similarity(query_vec, self.reference_matrix)[0]

    def is_in_domain(self, query: str) -> bool:
        return self._similarities(query).max() >= self.threshold

    def check(self, query: str):
        """Returns (in_domain: bool, fallback_message_or_none: str|None)"""
        if self.is_in_domain(query):
            return True, None
        return False, FALLBACK_MESSAGE

    def top_match(self, query: str):
        """Returns (best_faq_item, similarity_score) — the closest known FAQ
        entry to the query. Used by the API layer to show which handbook
        entry a generated answer was grounded against."""
        sims = self._similarities(query)
        best_idx = int(sims.argmax())
        return self.faq_items[best_idx], float(sims[best_idx])


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
