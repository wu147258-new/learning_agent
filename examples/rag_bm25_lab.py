"""Deterministic BM25 and retrieval-metric experiment.

No third-party dependencies, network access, file writes, or LLM calls.
All documents and query labels below are SYNTHETIC.
Run: python examples/rag_bm25_lab.py
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import re


TOKEN_PATTERN = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """English/code-term toy tokenizer; not a production Chinese tokenizer."""
    return TOKEN_PATTERN.findall(text.lower())


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    scope: str
    version: str


@dataclass(frozen=True)
class Hit:
    doc_id: str
    score: float
    version: str


class BM25Index:
    """Build an index over ONLY the documents available to a caller.

    The scope is a trusted server-side argument, never read from the query.
    Restricting indexed documents also avoids deriving corpus statistics from
    inaccessible documents. Rebuild on updates to keep versions consistent.
    """

    def __init__(self, docs: list[Document], allowed_scopes: frozenset[str],
                 k1: float = 1.2, b: float = 0.75):
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("k1 must be positive and b must be in [0,1]")
        if any(type(d) is not Document for d in docs):
            raise TypeError("expected Document records")
        visible = [d for d in docs if d.scope in allowed_scopes]
        ids = [d.doc_id for d in visible]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate visible doc_id")
        self.documents = {d.doc_id: d for d in visible}
        self.k1, self.b = k1, b
        self.tokens = {d.doc_id: tokenize(d.text) for d in visible}
        self.term_frequencies = {
            doc_id: Counter(words)
            for doc_id, words in self.tokens.items()
        }
        self.postings: dict[str, set[str]] = defaultdict(set)
        for doc_id, counts in self.term_frequencies.items():
            for term in counts:
                self.postings[term].add(doc_id)
        self.count = len(visible)
        self.avgdl = (
            sum(len(words) for words in self.tokens.values()) / self.count
            if self.count else 0.0
        )

    def idf(self, term: str) -> float:
        if not self.count:
            return 0.0
        df = len(self.postings.get(term, ()))
        if df == 0:
            return 0.0
        return math.log1p((self.count - df + 0.5) / (df + 0.5))

    def score(self, query: str, doc_id: str) -> float:
        if doc_id not in self.documents:
            raise KeyError("document is missing or not authorized")
        if self.avgdl == 0:
            return 0.0
        doc_length = len(self.tokens[doc_id])
        K = self.k1 * (1 - self.b + self.b * doc_length / self.avgdl)
        counts = self.term_frequencies[doc_id]
        result = 0.0
        for term in set(tokenize(query)):
            frequency = counts.get(term, 0)
            if frequency:
                result += self.idf(term) * (
                    frequency * (self.k1 + 1) / (frequency + K)
                )
        return result

    def search(self, query: str, top_k: int = 5) -> list[Hit]:
        if top_k < 0:
            raise ValueError("top_k must not be negative")
        terms = set(tokenize(query))
        candidate_ids = set().union(
            *(self.postings.get(term, set()) for term in terms)
        ) if terms else set()
        results = [
            Hit(doc_id=doc_id,
                score=self.score(query, doc_id),
                version=self.documents[doc_id].version)
            for doc_id in candidate_ids
        ]
        # Stable deterministic tie break. Exclude zero-score results.
        return sorted(
            (hit for hit in results if hit.score > 0),
            key=lambda h: (-h.score, h.doc_id)
        )[:top_k]


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant or k < 1:
        raise ValueError("relevant must be nonempty and k >= 1")
    return len(set(ranked[:k]) & relevant) / len(relevant)


def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if k < 1:
        raise ValueError("k must be >= 1")
    return len(set(ranked[:k]) & relevant) / k


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    for i, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            return 1 / i
    return 0.0


def mrr(ranking: dict[str, list[str]], labels: dict[str, set[str]]) -> float:
    if not labels:
        raise ValueError("at least one query required")
    if set(ranking) != set(labels):
        raise ValueError("ranking and relevance query IDs differ")
    return sum(
        reciprocal_rank(ranking[qid], labels[qid])
        for qid in labels
    ) / len(labels)


def dcg_at_k(ranked: list[str], grades: dict[str, int], k: int) -> float:
    if k < 1:
        raise ValueError("k must be >= 1")
    if any(type(v) is not int or v < 0 for v in grades.values()):
        raise ValueError("grades must be nonnegative integers")
    return sum(
        (2 ** grades.get(doc_id, 0) - 1) / math.log2(index + 1)
        for index, doc_id in enumerate(ranked[:k], start=1)
    )


def ndcg_at_k(ranked: list[str], grades: dict[str, int], k: int) -> float:
    ideal = sorted(grades, key=lambda d: (-grades[d], d))
    max_dcg = dcg_at_k(ideal, grades, k)
    return dcg_at_k(ranked, grades, k) / max_dcg if max_dcg else 0.0


def validate_evidence(index: BM25Index, hit: Hit, expected_version: str):
    """Reject missing, inaccessible or stale document evidence."""
    doc = index.documents.get(hit.doc_id)
    return doc is not None and doc.version == hit.version == expected_version


def self_test():
    docs = [
        Document("a", "jwt secret secret environment", "public", "v1"),
        Document("b", "jwt token signing header", "public", "v1"),
        Document("c", "database backup retention", "public", "v1"),
        Document("d", "python configuration files", "public", "v1"),
        Document("private", "jwt secret secret secret confidential", "admin", "v1"),
    ]
    public = BM25Index(docs, frozenset({"public"}))
    assert public.count == 4
    assert public.avgdl == 3.5
    assert len(public.postings["secret"]) == 1
    assert math.isclose(
        public.idf("secret"), math.log(10/3), abs_tol=1e-12
    )
    # For a four-token document, length differs from average, so test the
    # hand-calculation with a separate equal-length corpus.
    equal_docs = [
        Document("a", "jwt secret secret env", "public", "v1"),
        Document("b", "token signing auth test", "public", "v1"),
        Document("c", "database backup keep keys", "public", "v1"),
        Document("d", "python config files tests", "public", "v1"),
    ]
    equal_index = BM25Index(equal_docs, frozenset({"public"}))
    expected = math.log(10/3) * 2 * 2.2 / 3.2
    assert math.isclose(
        equal_index.score("secret", "a"), expected, abs_tol=1e-12
    )
    assert math.isclose(expected, 1.6555, abs_tol=0.0001)

    hits = public.search("jwt secret")
    assert hits[0].doc_id == "a"
    assert "private" not in [h.doc_id for h in hits]
    assert set(public.documents) == {"a", "b", "c", "d"}
    assert public.search("not-in-vocabulary") == []
    assert public.search("jwt", top_k=0) == []
    assert public.search("jwt") == public.search("jwt")

    # Changing the claimed role inside a query must not expand authorization.
    assert all(
        h.doc_id != "private"
        for h in public.search("admin role jwt secret")
    )
    try:
        public.score("secret", "private")
    except KeyError:
        pass
    else:
        raise AssertionError("unauthorized document unexpectedly scored")

    relevant = {"a", "b", "d"}
    ranked = ["a", "c", "b", "d"]
    assert math.isclose(recall_at_k(ranked, relevant, 3), 2/3)
    assert math.isclose(precision_at_k(ranked, relevant, 3), 2/3)
    assert reciprocal_rank(["c", "a", "b"], {"a"}) == 0.5
    assert mrr({"q1": ["a"], "q2": ["c", "b"]},
               {"q1": {"a"}, "q2": {"b"}}) == 0.75
    grades = {"a": 3, "b": 2, "c": 1, "d": 0}
    assert math.isclose(ndcg_at_k(["a", "b", "c"], grades, 3), 1)
    assert ndcg_at_k(["d", "c", "b"], grades, 3) < 1
    assert ndcg_at_k(["a"], {"a": 0}, 1) == 0

    assert validate_evidence(public, hits[0], "v1")
    changed = [Document(d.doc_id, d.text, d.scope, "v2")
               if d.doc_id == "a" else d for d in docs]
    updated = BM25Index(changed, frozenset({"public"}))
    assert not validate_evidence(updated, hits[0], "v2")
    new_hit = updated.search("jwt secret")[0]
    assert validate_evidence(updated, new_hit, "v2")

    for bad in (-1,):
        try:
            public.search("jwt", top_k=bad)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid k was accepted")

    print("PASS: BM25 hand-calculation, postings, scopes, metrics and versions")
    print(f"BM25 one-term contribution: {expected:.6f}")
    print("NOTE: Synthetic examples only; not a real-model or business benchmark.")


if __name__ == "__main__":
    self_test()
