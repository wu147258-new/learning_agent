"""Exact context packing with token budgets, authorization and evidence coverage.

The utility labels are SYNTHETIC, not semantic entailment predictions.
Complexity roughly O(n * budget * 2**num_required_evidence_types), so use this
dynamic program only for modest budgets and modest coverage sets.

Run: python examples/context_budget_lab.py
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256


@dataclass(frozen=True)
class Snippet:
    doc_id: str
    tenant: str
    version: str
    content: str
    tokens: int
    utility: int
    covers: frozenset[str]

    @property
    def digest(self) -> str:
        return sha256(self.content.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AuthorizedQuery:
    tenant: str
    corpus_version: str
    budget: int
    required_types: frozenset[str]


@dataclass(frozen=True)
class PackResult:
    selected: tuple[Snippet, ...]
    total_tokens: int
    total_utility: int
    has_complete_coverage: bool
    missing_types: frozenset[str]


def pack_context(candidates: list[Snippet], query: AuthorizedQuery):
    """Select evidence AFTER trusted tenant and version filtering.

    The caller cannot change the identity or version in the candidate.
    Production must derive AuthorizedQuery from a trusted service-side session.
    """
    if type(query.budget) is not int or query.budget < 0:
        raise ValueError("nonnegative integer budget required")
    if not query.tenant or not query.corpus_version:
        raise ValueError("trusted tenant and corpus version required")
    if len(query.required_types) > 12:
        raise ValueError("toy exact cover state limited to 12 types")

    visible = [
        d for d in candidates
        if d.tenant == query.tenant and d.version == query.corpus_version
    ]
    if len({d.doc_id for d in visible}) != len(visible):
        raise ValueError("duplicate visible source identity")
    for d in visible:
        if (type(d.tokens) is not int or type(d.utility) is not int or
                d.tokens < 1 or d.utility < 0):
            raise ValueError("positive token cost and nonnegative utility required")

    visible.sort(key=lambda d: d.doc_id)
    wanted = tuple(sorted(query.required_types))
    bit_for = {name: 1 << i for i, name in enumerate(wanted)}
    full_mask = (1 << len(wanted)) - 1

    # (used tokens, covered evidence mask) -> (utility, selected document IDs)
    states: dict[tuple[int, int], tuple[int, tuple[str, ...]]] = {
        (0, 0): (0, ())
    }
    lookup = {d.doc_id: d for d in visible}
    for doc in visible:
        mask = 0
        for evidence_type in doc.covers:
            mask |= bit_for.get(evidence_type, 0)
        updated = dict(states)
        for (used, covered), (utility, ids) in states.items():
            new_cost = used + doc.tokens
            if new_cost > query.budget:
                continue
            state = (new_cost, covered | mask)
            alternative = (utility + doc.utility, ids + (doc.doc_id,))
            old = updated.get(state)
            if (old is None or alternative[0] > old[0] or
                    (alternative[0] == old[0] and alternative[1] < old[1])):
                updated[state] = alternative
        states = updated

    complete = [
        (key, value) for key, value in states.items()
        if key[1] == full_mask
    ]
    eligible = complete if complete else list(states.items())
    # Prefer most covered evidence types, then higher utility, lower cost,
    # then lexicographically stable source IDs.
    best_key, best_value = min(
        eligible,
        key=lambda entry: (
            -entry[0][1].bit_count(),
            -entry[1][0], entry[0][0], entry[1][1]
        )
    )
    used, covered_mask = best_key
    _, selected_ids = best_value
    selected = tuple(lookup[d] for d in selected_ids)
    covered_names = frozenset(
        name for name, bit in bit_for.items() if covered_mask & bit
    )
    missing = query.required_types - covered_names
    assert sum(d.tokens for d in selected) == used
    assert sum(d.utility for d in selected) == best_value[0]
    return PackResult(
        selected, used, best_value[0], not missing, missing
    )


def verify_source(snippet: Snippet, expected_version: str,
                  expected_digest: str) -> bool:
    return (snippet.version == expected_version and
            snippet.digest == expected_digest)


def self_test():
    docs = [
        Snippet("A", "tenant-1", "v5", "long authentication excerpt",
                6, 10, frozenset({"auth"})),
        Snippet("B", "tenant-1", "v5", "short authentication excerpt",
                5, 8, frozenset({"auth"})),
        Snippet("C", "tenant-1", "v5", "short configuration excerpt",
                5, 8, frozenset({"config"})),
        # Higher ranking cannot bypass permissions or stale version.
        Snippet("private", "tenant-2", "v5", "private simulated evidence",
                1, 100, frozenset({"auth", "config"})),
        Snippet("stale", "tenant-1", "v4", "old simulated evidence",
                1, 100, frozenset({"auth", "config"})),
    ]
    query = AuthorizedQuery(
        "tenant-1", "v5", 10, frozenset({"auth", "config"})
    )
    result = pack_context(docs, query)
    assert tuple(d.doc_id for d in result.selected) == ("B", "C")
    assert result.total_tokens == 10
    assert result.total_utility == 16
    assert result.has_complete_coverage
    assert not result.missing_types
    assert all(d.tenant == "tenant-1" for d in result.selected)
    assert all(d.version == "v5" for d in result.selected)
    assert result == pack_context(list(reversed(docs)), query)

    # Too small a budget cannot prove a two-file dependency.
    limited = pack_context(
        docs, AuthorizedQuery(
            "tenant-1", "v5", 5, frozenset({"auth", "config"})
        )
    )
    assert not limited.has_complete_coverage
    assert limited.missing_types
    assert limited.total_tokens <= 5

    # Empty budget still produces a valid but explicitly insufficient pack.
    empty = pack_context(
        docs, AuthorizedQuery(
            "tenant-1", "v5", 0, frozenset({"auth"})
        )
    )
    assert empty.selected == ()
    assert empty.missing_types == frozenset({"auth"})

    for snippet in result.selected:
        assert verify_source(snippet, snippet.version, snippet.digest)
        assert not verify_source(snippet, "stale", snippet.digest)
        assert not verify_source(snippet, snippet.version, "incorrect-hash")

    try:
        pack_context(
            docs, AuthorizedQuery(
                "tenant-1", "v5", -1, frozenset({"auth"})
            )
        )
    except ValueError:
        pass
    else:
        raise AssertionError("negative budget should be rejected")

    print("PASS context: exact knapsack optimum B+C utility=16")
    print("PASS context: tenant, version, evidence coverage and integrity")
    print("NOTE: relevance utilities are synthetic; no model quality benchmark.")


if __name__ == "__main__":
    self_test()
