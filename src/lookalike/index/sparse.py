"""BM25 sparse vectors computed in the repo (DECISIONS.md D21).

Document weight of term t in document d:
    idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(d) / avg_len))
    idf(t) = ln((N - n_t + 0.5) / (n_t + 0.5) + 1)
The query vector has weight 1 for each distinct known term, so dot(query, doc) is exactly the
BM25 score. The same weights are uploaded to Qdrant, so offline and Qdrant scores match.
"""

import math
import re
from collections import Counter

import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9]+")
# Small, fixed list: function words only. Fashion words ("with", "and") carry little signal.
STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to with".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens without stopwords. No stemming."""
    if not isinstance(text, str):
        return []
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


class BM25:
    """Fit on the documents of a gallery, then produce sparse document and query vectors."""

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def fit(self, docs: list[list[str]]) -> "BM25":
        if not docs:
            raise ValueError("cannot fit BM25 on zero documents")
        n_docs = len(docs)
        df = Counter(t for doc in docs for t in set(doc))
        self.vocab = {t: i for i, t in enumerate(sorted(df))}
        n_t = np.array([df[t] for t in sorted(df)], dtype=np.float64)
        self.idf = np.log((n_docs - n_t + 0.5) / (n_t + 0.5) + 1.0)
        lengths = [len(d) for d in docs]
        self.avg_len = sum(lengths) / n_docs if sum(lengths) else 1.0
        return self

    def doc_vector(self, doc: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """(indices, values) of one document; unknown terms are ignored."""
        tf = Counter(t for t in doc if t in self.vocab)
        if not tf:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        pairs = sorted((self.vocab[t], count) for t, count in tf.items())
        idx = np.array([i for i, _ in pairs], dtype=np.int64)
        freq = np.array([c for _, c in pairs], dtype=np.float64)
        norm = self.k1 * (1 - self.b + self.b * len(doc) / self.avg_len)
        values = self.idf[idx] * freq * (self.k1 + 1) / (freq + norm)
        return idx, values.astype(np.float32)

    def query_vector(self, doc: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """(indices, ones) for each distinct known term of the query."""
        idx = np.array(sorted({self.vocab[t] for t in doc if t in self.vocab}), dtype=np.int64)
        return idx, np.ones(len(idx), dtype=np.float32)

    def dense_matrix(self, docs: list[list[str]], query: bool = False) -> np.ndarray:
        """(len(docs), vocab_size) float32 matrix, for exact scoring of a small split."""
        out = np.zeros((len(docs), len(self.vocab)), dtype=np.float32)
        vec = self.query_vector if query else self.doc_vector
        for row, doc in enumerate(docs):
            idx, values = vec(doc)
            out[row, idx] = values
        return out

    def score_matrix(
        self, query_docs: list[list[str]], gallery_docs: list[list[str]]
    ) -> np.ndarray:
        """BM25 score of every gallery document for every query: (n_queries, n_gallery)."""
        return self.dense_matrix(query_docs, query=True) @ self.dense_matrix(gallery_docs).T


def math_bm25(query: list[str], doc: list[str], docs: list[list[str]], k1=1.2, b=0.75) -> float:
    """Textbook BM25 written term by term. Used only by tests to check the vectorised version."""
    n = len(docs)
    avg = sum(len(d) for d in docs) / n
    score = 0.0
    for term in set(query):
        n_t = sum(term in d for d in docs)
        if n_t == 0:
            continue
        idf = math.log((n - n_t + 0.5) / (n_t + 0.5) + 1)
        tf = doc.count(term)
        score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(doc) / avg))
    return score
