"""
Retrieval: given an incoming customer message, find the most similar
historically-resolved threads so the reply drafter can ground its answer
in how this brand actually responded before (rather than hallucinating
a generic answer).

Approach: TF-IDF + cosine similarity over customer_text, restricted to a
"clean" historical pool (see build_index). This is deliberately simple
(no embeddings API call needed) -- see decision_log.md for why.
"""

import json
from pathlib import Path
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = Path(__file__).parent.parent / "data" / "raw" / "threads.jsonl"


@dataclass
class HistoricalMatch:
    thread_id: int
    customer_text: str
    historical_response_text: str
    true_intent: str
    similarity: float


class HistoricalIndex:
    def __init__(self, data_path: Path = DATA_PATH, holdout_thread_ids=None):
        """
        holdout_thread_ids: set of thread_ids to EXCLUDE from the index.
        This must be used to exclude golden-eval-set threads, otherwise
        retrieval would "cheat" by retrieving the exact eval example's
        own historical answer -- see report/REPORT.md 'misleading number'
        section for why this matters.
        """
        self.threads = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if holdout_thread_ids and rec["thread_id"] in holdout_thread_ids:
                    continue
                self.threads.append(rec)

        self.texts = [t["customer_text"] for t in self.threads]
        self.vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
        self.matrix = self.vectorizer.fit_transform(self.texts)

    def search(self, query: str, k: int = 3) -> list[HistoricalMatch]:
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.matrix)[0]
        top_idx = sims.argsort()[::-1][:k]
        results = []
        for idx in top_idx:
            t = self.threads[idx]
            results.append(
                HistoricalMatch(
                    thread_id=t["thread_id"],
                    customer_text=t["customer_text"],
                    historical_response_text=t["historical_response_text"],
                    true_intent=t["true_intent"],
                    similarity=float(sims[idx]),
                )
            )
        return results


if __name__ == "__main__":
    idx = HistoricalIndex()
    test_query = "my iphone battery is draining super fast after the update"
    matches = idx.search(test_query, k=3)
    print(f"Query: {test_query}\n")
    for m in matches:
        print(f"[sim={m.similarity:.3f}] intent={m.true_intent}")
        print(f"  customer: {m.customer_text}")
        print(f"  resolved: {m.historical_response_text}\n")
