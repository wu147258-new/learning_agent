"""Unigram segmentation with a Viterbi dynamic program, standard library only."""
from __future__ import annotations
import math

VOCAB = {"a": 0.1, "b": 0.3, "ab": 0.2, "ba": 0.4}


def optimal(text: str, vocab: dict[str, float]):
    if not all(t and 0 < p <= 1 for t, p in vocab.items()):
        raise ValueError("invalid token probabilities")
    n = len(text)
    best = [-math.inf]*(n+1)
    back = [None]*(n+1)
    best[0] = 0.0
    for i in range(1, n+1):
        for j in range(i):
            piece = text[j:i]
            if piece in vocab:
                score = best[j]+math.log(vocab[piece])
                if score > best[i]:
                    best[i], back[i] = score, (j, piece)
    if back[n] is None and n:
        raise ValueError("text cannot be segmented")
    parts = []
    i = n
    while i:
        j, piece = back[i]
        parts.append(piece)
        i = j
    parts.reverse()
    assert "".join(parts) == text
    return parts, best[n]


def greedy(text, vocab):
    pieces = []
    while text:
        candidates = [v for v in vocab if text.startswith(v)]
        if not candidates:
            raise ValueError("greedy dead end")
        v = max(candidates, key=lambda s: (len(s), s))
        pieces.append(v)
        text = text[len(v):]
    return pieces


def main():
    tokens, score = optimal("aba", VOCAB)
    assert tokens == ["a", "ba"], tokens
    assert math.isclose(math.exp(score), 0.04, abs_tol=1e-12)
    greed = greedy("aba", VOCAB)
    assert greed == ["ab", "a"]
    assert math.isclose(math.prod(VOCAB[t] for t in greed), 0.02)
    for sample in ["a", "b", "ab", "ba", "aba", "abab", "baba"]:
        items, _ = optimal(sample, VOCAB)
        assert "".join(items) == sample
    for bad in ["xyz", "cab"]:
        try:
            optimal(bad, VOCAB)
        except ValueError:
            pass
        else:
            raise AssertionError(f"should reject {bad!r}")
    print("PASS Viterbi: a+ba=0.04 vs longest-match ab+a=0.02")
    print("PASS reconstruction invariants and impossible strings")


if __name__ == "__main__":
    main()
