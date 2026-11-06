"""Hybrid retrieval: lexical BM25, deterministic toy vectors, ACL and evidence.

No pretrained embeddings, no HTTP and no external database. The vector
function is a reproducible hash feature map, NOT semantic understanding.
Run: python3 examples/hybrid_retrieval_acl_lab.py
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
from math import log, sqrt
import re
from typing import Iterable


@dataclass(frozen=True)
class Document:
    doc_id: str
    tenant: str
    scope: str
    version: int
    title: str
    text: str
    expiry: int

    def fingerprint(self):
        return sha256(
            (self.doc_id + "\x00" + self.text + "\x00"
             + str(self.version)).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class User:
    tenant: str
    scopes: frozenset[str]


@dataclass(frozen=True)
class Evidence:
    doc_id: str
    version: int
    start: int
    end: int
    quote: str
    digest: str


def tokenize(text: str) -> list[str]:
    """Mixed toy tokenizer: Latin words and overlapping CJK bigrams."""
    parts = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+", text.lower())
    tokens = []
    for piece in parts:
        if re.fullmatch(r"[\u4e00-\u9fff]+", piece):
            if len(piece) == 1:
                tokens.append(piece)
            else:
                tokens.extend(piece[i:i+2] for i in range(len(piece)-1))
        else:
            tokens.append(piece)
    return tokens


def vec(text: str, dim: int = 32) -> list[float]:
    if dim <= 0:
        raise ValueError("dimension must be positive")
    counts = Counter(tokenize(text))
    v = [0.0] * dim
    for term, weight in counts.items():
        digest = sha256(term.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % dim
        sign = 1 if digest[4] % 2 == 0 else -1
        v[idx] += sign * weight
    norm = sqrt(sum(z*z for z in v))
    return [x / norm for x in v] if norm else v


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("dimension mismatch")
    return sum(x*y for x, y in zip(a, b))


def bm25_score(
    query_tokens: list[str],
    doc_tokens: list[str],
    doc_freq: Counter,
    n_docs: int,
    avg_length: float,
    k1: float = 1.2,
    b: float = 0.75,
) -> float:
    if not n_docs or not avg_length:
        return 0.0
    counts = Counter(doc_tokens)
    score = 0.0
    for term in set(query_tokens):
        n = doc_freq[term]
        idf = log(1 + (n_docs - n + 0.5)/(n + 0.5))
        tf = counts[term]
        if not tf:
            continue
        length_factor = k1 * (1 - b + b * len(doc_tokens)/avg_length)
        score += idf * (tf * (k1+1))/(tf + length_factor)
    return score


def permitted(user: User, doc: Document, now: int) -> bool:
    return (user.tenant == doc.tenant and
            doc.scope in user.scopes and doc.expiry > now and
            doc.version >= 1)


def reciprocal_rank_fusion(rankings: list[list[str]], k=60):
    if k <= 0:
        raise ValueError("k must be positive")
    total: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        if len(ranking) != len(set(ranking)):
            raise ValueError("duplicate IDs in one ranking")
        for idx, doc_id in enumerate(ranking, start=1):
            total[doc_id] += 1/(k + idx)
    return sorted(total, key=lambda x: (-total[x], x))


class Index:
    def __init__(self, docs: Iterable[Document]):
        docs = list(docs)
        if len({d.doc_id for d in docs}) != len(docs):
            raise ValueError("document IDs must be unique")
        self.docs = {d.doc_id: d for d in docs}

    def allowed(self, user: User, now: int) -> list[Document]:
        return sorted(
            (doc for doc in self.docs.values()
             if permitted(user, doc, now)),
            key=lambda d: d.doc_id
        )

    def search(self, query: str, user: User, now: int, top_k=3):
        if top_k < 1:
            raise ValueError("top_k must be positive")
        # Authorization BEFORE scoring, RRF, re-ranking, or returning text.
        allowed = self.allowed(user, now)
        if not allowed:
            return []
        query_tokens = tokenize(query)
        tokens_by_id = {
            d.doc_id: tokenize(d.title + " " + d.text) for d in allowed
        }
        n = len(allowed)
        average = sum(len(tokens_by_id[d.doc_id]) for d in allowed)/n
        df = Counter()
        for tokens in tokens_by_id.values():
            df.update(set(tokens))
        lexical = {}
        vectorial = {}
        qvec = vec(query)
        for d in allowed:
            lexical[d.doc_id] = bm25_score(
                query_tokens, tokens_by_id[d.doc_id], df, n, average
            )
            vectorial[d.doc_id] = cosine(
                qvec, vec(d.title + " " + d.text)
            )
        lex_rank = sorted(
            lexical, key=lambda x: (-lexical[x], x)
        )
        vec_rank = sorted(
            vectorial, key=lambda x: (-vectorial[x], x)
        )
        result = reciprocal_rank_fusion([lex_rank, vec_rank])
        return [(doc_id, lexical[doc_id], vectorial[doc_id])
                for doc_id in result[:top_k]]

    def citation(self, user: User, doc_id: str,
                 snippet: str, now: int) -> Evidence:
        doc = self.docs.get(doc_id)
        if doc is None or not permitted(user, doc, now):
            raise PermissionError("no access to cited document")
        idx = doc.text.find(snippet)
        if idx < 0:
            raise ValueError("quote absent from document")
        return Evidence(
            doc.doc_id, doc.version, idx, idx + len(snippet),
            snippet, doc.fingerprint(),
        )

    def verify(self, user: User, evidence: Evidence, now: int):
        doc = self.docs.get(evidence.doc_id)
        if doc is None or not permitted(user, doc, now):
            return False
        if doc.version != evidence.version:
            return False
        if doc.fingerprint() != evidence.digest:
            return False
        return (0 <= evidence.start <= evidence.end <= len(doc.text)
                and doc.text[evidence.start:evidence.end] == evidence.quote)


def precision_at_k(ranking: list[str], relevant: set[str], k: int):
    if k <= 0:
        raise ValueError("k must be positive")
    return len(set(ranking[:k]).intersection(relevant)) / k


def recall_at_k(ranking: list[str], relevant: set[str], k: int):
    if not relevant:
        raise ValueError("relevant set must be nonempty")
    return len(set(ranking[:k]).intersection(relevant)) / len(relevant)


def reciprocal_rank(ranking: list[str], relevant: set[str]):
    for i, doc_id in enumerate(ranking, start=1):
        if doc_id in relevant:
            return 1/i
    return 0.0


def dcg(grades: list[int], k: int):
    return sum((2**grade - 1)/log(i + 2, 2)
               for i, grade in enumerate(grades[:k]))


def ndcg_at_k(ranking, grade_map, k):
    got = dcg([grade_map.get(x, 0) for x in ranking], k)
    ideal = dcg(sorted(grade_map.values(), reverse=True), k)
    return got/ideal if ideal else 0.0


def test_scores():
    docs = [["alpha", "alpha", "beta"], ["beta", "gamma"]]
    df = Counter()
    for items in docs:
        df.update(set(items))
    score_a = bm25_score(["alpha"], docs[0], df, 2, 2.5)
    score_b = bm25_score(["alpha"], docs[1], df, 2, 2.5)
    assert score_a > score_b == 0
    assert tokenize("数据安全") == ["数据", "据安", "安全"]
    assert abs(cosine(vec("same text"), vec("same text")) - 1) < 1e-12
    rrf = reciprocal_rank_fusion([
        ["d1", "d2", "d3"], ["d2", "d1", "d3"]
    ])
    assert rrf[:2] == ["d1", "d2"]  # exact tie resolved by ID
    assert precision_at_k(["a", "b", "c"], {"a", "c"}, 2) == 0.5
    assert recall_at_k(["a", "b", "c"], {"a", "c"}, 2) == 0.5
    assert reciprocal_rank(["x", "a"], {"a"}) == 0.5
    assert ndcg_at_k(["a", "b"], {"a": 2, "b": 1}, 2) == 1.0
    print("PASS BM25, deterministic vector, fusion and ranking metrics")


def synthetic_docs():
    return [
        Document("d1", "A", "alerts:read", 1, "DNS", 
                 "DNS异常连接 需要核对授权节点和域名证据", 999),
        Document("d2", "A", "alerts:read", 1, "HTTP",
                 "HTTP接口出现授权失败和请求时间异常", 999),
        Document("d3", "B", "alerts:read", 1, "DNS",
                 "DNS机密数据 不允许跨租户检索", 999),
        Document("d4", "A", "alerts:read", 1, "expired",
                 "过期告警 凭据已经失效", 99),
        Document("d5", "A", "admin:read", 1, "secret",
                 "管理员密钥不能由普通检索暴露", 999),
    ]


def test_acl_citation():
    user = User("A", frozenset({"alerts:read"}))
    ix = Index(synthetic_docs())
    results = ix.search("DNS授权节点", user, now=100)
    ids = [x[0] for x in results]
    assert set(ids).issubset({"d1", "d2"})
    assert ids[0] == "d1", ids
    evidence = ix.citation(user, "d1", "授权节点", now=100)
    assert ix.verify(user, evidence, now=100)
    tampered = Evidence(
        evidence.doc_id, evidence.version, evidence.start,
        evidence.end, "篡改内容", evidence.digest
    )
    assert not ix.verify(user, tampered, now=100)
    ix.docs["d1"] = Document(
        "d1", "A", "alerts:read", 2, "DNS",
        "更新后该版本的内容已经改变", 999
    )
    assert not ix.verify(user, evidence, now=100)
    try:
        ix.citation(user, "d3", "机密", now=100)
    except PermissionError:
        pass
    else:
        raise AssertionError("tenant isolation failed")
    print("PASS authorization-before-scoring, citations and version revocation")


if __name__ == "__main__":
    test_scores()
    test_acl_citation()
    print("NOTE: hash vectors are NOT semantic embeddings or an ANN index.")
