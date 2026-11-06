"""Offline SFT data and mask lab with deterministic, inspectable examples.

This module is not a full tokenizer nor a real LM training run. It tests
data provenance, group splits, label masks, boundaries and stable loss.
Run: python3 examples/sft_data_loss_pipeline.py
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import exp, log
import random
import re
import unicodedata


@dataclass(frozen=True)
class Record:
    document_id: str
    group_id: str
    license: str
    instruction: str
    answer: str


@dataclass(frozen=True)
class Sample:
    document_id: str
    group_id: str
    tokens: tuple[int, ...]
    labels: tuple[int, ...]
    loss_mask: tuple[int, ...]


def normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).strip()


def fingerprint(record: Record) -> str:
    body = normalize(record.instruction) + "\n" + normalize(record.answer)
    return sha256(body.encode("utf-8")).hexdigest()


def sanitize(records: list[Record], allowed_licenses: set[str]):
    seen_hashes = set()
    kept = []
    rejected = []
    for r in records:
        if not r.document_id or not r.group_id:
            rejected.append((r.document_id, "missing metadata"))
            continue
        if r.license not in allowed_licenses:
            rejected.append((r.document_id, "license not allowed"))
            continue
        if not normalize(r.instruction) or not normalize(r.answer):
            rejected.append((r.document_id, "empty content"))
            continue
        h = fingerprint(r)
        if h in seen_hashes:
            rejected.append((r.document_id, "exact duplicate"))
            continue
        seen_hashes.add(h)
        kept.append(r)
    return kept, rejected


class ToyTokenizer:
    """Word-and-punctuation tokenizer with fixed vocabulary."""

    def __init__(self, corpus: list[str]):
        self.special = ["<pad>", "<bos>", "<eos>", "<unk>",
                        "<user>", "<assistant>"]
        words = sorted({
            token.lower()
            for text in corpus
            for token in self.split(text)
        })
        self.vocab = {x: i for i, x in enumerate(
            self.special + [w for w in words if w not in self.special]
        )}
        self.reverse = {i: x for x, i in self.vocab.items()}

    @staticmethod
    def split(text: str) -> list[str]:
        return re.findall(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]", text)

    def encode(self, text: str) -> list[int]:
        return [self.vocab.get(x.lower(), self.vocab["<unk>"])
                for x in self.split(text)]

    def decode(self, ids: list[int]) -> str:
        return " ".join(self.reverse.get(i, "<bad>") for i in ids)


def encode_supervised(r: Record, tok: ToyTokenizer) -> Sample:
    bos = tok.vocab["<bos>"]
    usr = tok.vocab["<user>"]
    ast = tok.vocab["<assistant>"]
    eos = tok.vocab["<eos>"]
    prompt = [bos, usr] + tok.encode(normalize(r.instruction)) + [ast]
    completion = tok.encode(normalize(r.answer)) + [eos]
    tokens = tuple(prompt + completion)
    mask = tuple([0] * len(prompt) + [1] * len(completion))
    labels = tuple(-100 if not train else tokens[i]
                   for i, train in enumerate(mask))
    return Sample(r.document_id, r.group_id, tokens, labels, mask)


def split_by_group(records: list[Record], seed=23,
                   fraction_train=0.75):
    if not 0 < fraction_train < 1:
        raise ValueError("split fraction must be strictly between 0 and 1")
    groups = sorted({r.group_id for r in records})
    rng = random.Random(seed)
    rng.shuffle(groups)
    if len(groups) < 2:
        raise ValueError("at least two groups needed")
    count = max(1, min(len(groups)-1,
                       round(len(groups) * fraction_train)))
    train_groups = set(groups[:count])
    train = [r for r in records if r.group_id in train_groups]
    test = [r for r in records if r.group_id not in train_groups]
    assert {r.group_id for r in train}.isdisjoint(
        {r.group_id for r in test}
    )
    return train, test


def pack(samples: list[Sample], capacity: int):
    """Pack without crossing sample/document boundaries in attention."""
    if capacity < 2:
        raise ValueError("capacity too small")
    batches = []
    current = []
    for sample in samples:
        if len(sample.tokens) > capacity:
            raise ValueError("sample exceeds context")
        if sum(len(x.tokens) for x in current) + len(sample.tokens) > capacity:
            batches.append(current)
            current = []
        current.append(sample)
    if current:
        batches.append(current)
    result = []
    for batch in batches:
        tokens, labels, mask, segments = [], [], [], []
        for part_no, sample in enumerate(batch):
            tokens.extend(sample.tokens)
            labels.extend(sample.labels)
            mask.extend(sample.loss_mask)
            segments.extend([part_no] * len(sample.tokens))
        attention = [
            [int(j <= i and segments[j] == segments[i])
             for j in range(len(tokens))]
            for i in range(len(tokens))
        ]
        result.append({
            "tokens": tokens, "labels": labels, "mask": mask,
            "segments": segments, "attention": attention,
        })
    return result


def logsumexp(values: list[float]) -> float:
    if not values:
        raise ValueError("empty logits")
    m = max(values)
    return m + log(sum(exp(x-m) for x in values))


def cross_entropy(logits: list[list[float]],
                  targets: list[int], ignore_index=-100) -> float:
    if len(logits) != len(targets):
        raise ValueError("sequence shape mismatch")
    total, count = 0.0, 0
    for row, target in zip(logits, targets):
        if target == ignore_index:
            continue
        if not 0 <= target < len(row):
            raise ValueError("label outside vocabulary")
        total += logsumexp(row) - row[target]
        count += 1
    if count == 0:
        raise ValueError("empty supervised target")
    return total / count


def lora_count(input_dim: int, output_dim: int, rank: int):
    if min(input_dim, output_dim, rank) <= 0:
        raise ValueError("positive dimensions required")
    if rank > min(input_dim, output_dim):
        raise ValueError("rank is not low")
    return rank * (input_dim + output_dim)


def matmul(a: list[list[float]], b: list[list[float]]):
    if not a or not b or len(a[0]) != len(b):
        raise ValueError("shape mismatch")
    if len({len(row) for row in a}) > 1:
        raise ValueError("jagged matrix")
    if len({len(row) for row in b}) > 1:
        raise ValueError("jagged matrix")
    return [[sum(a[i][k] * b[k][j] for k in range(len(b)))
             for j in range(len(b[0]))] for i in range(len(a))]


def rank_one_delta(b: list[list[float]], a: list[list[float]]):
    if not b or not a or len(b[0]) != 1 or len(a) != 1:
        raise ValueError("rank one factors needed")
    return matmul(b, a)


def training_records():
    # The words are intentionally synthetic; not a user dataset.
    return [
        Record("d01", "topic-1", "CC0", "find x in x + 1 = 3", "x = 2"),
        Record("d02", "topic-1", "CC0", "find y in y + 2 = 4", "y = 2"),
        Record("d03", "topic-2", "CC0", "define a state", "a condition"),
        Record("d04", "topic-3", "CC0", "add 2 and 5", "7"),
        Record("d05", "topic-4", "CC0", "multiply 3 by 4", "12"),
        Record("d06", "topic-5", "CC0", "list two colors", "red blue"),
        Record("d07", "topic-6", "CC0", "reverse ab", "ba"),
        Record("d08", "topic-6", "CC0", "reverse cd", "dc"),
        Record("d09", "topic-7", "restricted", "private content", "secret"),
        Record("d10", "topic-8", "CC0", "add 2 and 5", "7"),
        Record("d11", "topic-9", "CC0", "  add 2 and 5  ", "7"),
    ]


def test_dataset():
    kept, dropped = sanitize(training_records(), {"CC0"})
    assert len(kept) == 8
    assert len(dropped) == 3
    assert {reason for _, reason in dropped} == {
        "license not allowed", "exact duplicate"
    }
    train, test = split_by_group(kept)
    assert train and test
    train2, test2 = split_by_group(kept)
    assert train == train2 and test == test2
    print("PASS license, normalization, duplicate and group split")


def test_mask_packing():
    records, _ = sanitize(training_records(), {"CC0"})
    tok = ToyTokenizer([r.instruction for r in records] +
                       [r.answer for r in records])
    samples = [encode_supervised(r, tok) for r in records]
    for sample in samples:
        assert sum(sample.loss_mask) > 0
        assert all(y == -100 for y, m in zip(sample.labels,
                                             sample.loss_mask) if m == 0)
        assert all(y == sample.tokens[i]
                   for i, (y, m) in enumerate(zip(
                       sample.labels, sample.loss_mask)) if m == 1)
    batches = pack(samples, 50)
    assert len(batches) >= 2
    for batch in batches:
        assert len(batch["tokens"]) <= 50
        for i, row in enumerate(batch["attention"]):
            assert row[i] == 1
            assert all(row[j] == 0 for j in range(i + 1, len(row)))
            assert all(row[j] == 0 for j in range(i)
                       if batch["segments"][i] != batch["segments"][j])
    print("PASS answer-only target masking and segment-aware causal packing")


def test_loss_and_lora():
    logits = [[-1000.0, 1000.0, 1.0],
              [1000.0, 999.0, 998.0]]
    targets = [-100, 0]
    expected = log(1 + exp(-1) + exp(-2))
    assert abs(cross_entropy(logits, targets) - expected) < 1e-12
    assert lora_count(64, 32, 4) == 384
    delta = rank_one_delta([[2.0], [3.0]], [[4.0, 5.0]])
    assert delta == [[8.0, 10.0], [12.0, 15.0]]
    print("PASS stable answer-token CE and LoRA shapes")


def main():
    test_dataset()
    test_mask_packing()
    test_loss_and_lora()
    print("NOTE: data is synthetic; model quality has not been evaluated.")


if __name__ == "__main__":
    main()
