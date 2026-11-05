"""End-to-end SFT then DPO on an actual tiny Transformer decoder.

PyTorch CPU, no model downloads, no API keys, synthetic sequences only.
Run: python examples/tiny_lm_sft_dpo.py

The goal is to validate response-only token likelihood and a frozen DPO
reference on a runnable autoregressive model. This is NOT generalization.
"""
from __future__ import annotations

from copy import deepcopy
import math

import torch
import torch.nn.functional as F

from tiny_transformer import TinyDecoder


VOCAB = 16
PROMPT_LENGTH = 2


def response_log_probs(model: TinyDecoder, sequences: torch.Tensor):
    """Log P(response | prompt), NOT log P(prompt+response)."""
    if sequences.ndim != 2 or sequences.shape[1] <= PROMPT_LENGTH:
        raise ValueError("expected batched prompt and response tokens")
    # Input at position t predicts sequence[t+1]. First answer token is
    # sequence[PROMPT_LENGTH], predicted at input position PROMPT_LENGTH-1.
    logits = model(sequences[:, :-1])
    labels = sequences[:, 1:]
    token_logp = F.log_softmax(logits, dim=-1).gather(
        -1, labels.unsqueeze(-1)
    ).squeeze(-1)
    mask = (
        torch.arange(labels.shape[1], device=sequences.device)
        >= PROMPT_LENGTH - 1
    )
    assert mask.sum().item() == sequences.shape[1] - PROMPT_LENGTH
    return token_logp[:, mask].sum(dim=-1)


def response_sft_loss(model, sequences):
    return -response_log_probs(model, sequences).sum() / (
        sequences.shape[0] * (sequences.shape[1] - PROMPT_LENGTH)
    )


def dpo_pair_loss(
    model, positive, negative, reference_positive, reference_negative,
    beta=0.5
):
    if beta <= 0:
        raise ValueError("beta must be positive")
    chosen = response_log_probs(model, positive)
    rejected = response_log_probs(model, negative)
    # Reference scores are computed from a frozen snapshot, without gradient.
    margin = (chosen - rejected) - (
        reference_positive.detach() - reference_negative.detach()
    )
    return -F.logsigmoid(beta * margin).mean()


def build_data():
    # Format: [BOS, task-id, answer-token, answer-token, EOS]
    # Two task prompts with preferred and rejected synthetic answers.
    chosen = torch.tensor([
        [1, 2, 5, 6, 15],
        [1, 3, 7, 8, 15],
    ], dtype=torch.long)
    rejected = torch.tensor([
        [1, 2, 9, 10, 15],
        [1, 3, 11, 12, 15],
    ], dtype=torch.long)
    return chosen, rejected


def self_test():
    torch.set_num_threads(1)
    torch.manual_seed(2026)
    chosen, rejected = build_data()
    model = TinyDecoder(vocab=VOCAB, width=24, heads=4,
                        layers=1, max_length=8)

    # Step 1: SFT on ONLY response positions.
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01,
                                   weight_decay=0)
    model.eval()  # Deterministic forward passes for this toy experiment.
    initial_sft = response_sft_loss(model, chosen).item()
    for _ in range(75):
        optimizer.zero_grad(set_to_none=True)
        loss = response_sft_loss(model, chosen)
        loss.backward()
        optimizer.step()
    final_sft = response_sft_loss(model, chosen).item()
    assert final_sft < initial_sft * 0.6, (initial_sft, final_sft)

    # Check the log-likelihood mask by manual summation.
    logits = model(chosen[:, :-1])
    labels = chosen[:, 1:]
    token_logp = F.log_softmax(logits, -1).gather(
        -1, labels.unsqueeze(-1)
    ).squeeze(-1)
    manual = token_logp[:, PROMPT_LENGTH-1:].sum(-1)
    assert torch.allclose(response_log_probs(model, chosen), manual,
                          atol=1e-6)

    # Step 2: snapshot and freeze an SFT reference.
    reference = deepcopy(model)
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        ref_pos = response_log_probs(reference, chosen)
        ref_neg = response_log_probs(reference, rejected)

    # Step 3: DPO optimization with reference held constant.
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003,
                                   weight_decay=0)
    initial_dpo = dpo_pair_loss(
        model, chosen, rejected, ref_pos, ref_neg
    ).item()
    assert math.isclose(initial_dpo, math.log(2), abs_tol=1e-6)
    for _ in range(45):
        optimizer.zero_grad(set_to_none=True)
        loss = dpo_pair_loss(model, chosen, rejected, ref_pos, ref_neg)
        loss.backward()
        optimizer.step()
    final_dpo = dpo_pair_loss(
        model, chosen, rejected, ref_pos, ref_neg
    ).item()
    assert final_dpo < initial_dpo * 0.75, (initial_dpo, final_dpo)
    assert all(parameter.grad is None for parameter in reference.parameters())

    current_margin = (
        response_log_probs(model, chosen) -
        response_log_probs(model, rejected)
    ).detach()
    ref_margin = ref_pos - ref_neg
    assert torch.all(current_margin > ref_margin)

    print(f"PASS tiny Decoder SFT answer-token CE: {initial_sft:.4f} -> {final_sft:.4f}")
    print(f"PASS tiny Decoder DPO preference loss: {initial_dpo:.4f} -> {final_dpo:.4f}")
    print("PASS reference frozen, prompt excluded, token alignment verified")
    print("NOTE: Memorizing two synthetic prompts is not instruction following.")


if __name__ == "__main__":
    self_test()
