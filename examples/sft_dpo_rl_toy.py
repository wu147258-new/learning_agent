"""SFT, DPO and expected-reward gradients on a two-action model.

Only synthetic labels, preferences and rewards. No pretrained weights or APIs.
Requires PyTorch 2.x CPU. This is not a real LLM benchmark.
"""
import torch
from torch.nn import functional as F


def fit_sft():
    logits = torch.nn.Parameter(torch.zeros(2))
    optimizer = torch.optim.SGD([logits], lr=0.25)
    labels = torch.tensor([0] * 8 + [1] * 2)
    before = F.cross_entropy(logits.expand(10, -1), labels).item()
    for _ in range(160):
        optimizer.zero_grad()
        loss = F.cross_entropy(logits.expand(10, -1), labels)
        loss.backward()
        optimizer.step()
    probs = F.softmax(logits.detach(), dim=0)
    after = F.cross_entropy(logits.expand(10, -1), labels).item()
    assert after < before and abs(probs[0].item() - 0.8) < 0.02
    return before, after, probs


def fit_dpo():
    chosen, rejected, beta = 1, 0, 0.75
    logits = torch.nn.Parameter(torch.zeros(2))
    reference = torch.tensor([0.0, 0.0])
    optimizer = torch.optim.SGD([logits], lr=0.2)

    def loss_fn():
        log_probs = F.log_softmax(logits, dim=-1)
        log_ref = F.log_softmax(reference, dim=-1)
        difference = ((log_probs[chosen] - log_probs[rejected])
                      - (log_ref[chosen] - log_ref[rejected]))
        return -F.logsigmoid(beta * difference)

    before = loss_fn().item()
    for _ in range(70):
        optimizer.zero_grad()
        loss = loss_fn()
        loss.backward()
        optimizer.step()
    after = loss_fn().item()
    probs = F.softmax(logits.detach(), dim=0)
    assert after < before and probs[chosen].item() > 0.5
    return before, after, probs


def fit_expected_reward():
    logits = torch.nn.Parameter(torch.zeros(2))
    rewards = torch.tensor([1.0, 3.0])
    optimizer = torch.optim.SGD([logits], lr=0.2)

    probs = F.softmax(logits, dim=0)
    expectation = (probs * rewards).sum()
    expectation.backward()
    direct_gradient = logits.grad.detach().clone()
    logits.grad.zero_()

    # Baseline term is independent of the sampled action.
    baseline = expectation.detach()
    log_probs = F.log_softmax(logits, dim=0)
    score_function = (probs.detach() * (rewards - baseline) * log_probs).sum()
    score_function.backward()
    score_gradient = logits.grad.detach().clone()
    assert torch.allclose(direct_gradient, score_gradient, atol=1e-6)
    assert direct_gradient[0].item() < 0 < direct_gradient[1].item()

    logits.grad = None
    for _ in range(60):
        optimizer.zero_grad()
        expectation = (F.softmax(logits, dim=0) * rewards).sum()
        (-expectation).backward()
        optimizer.step()

    final_probs = F.softmax(logits.detach(), dim=0)
    final_expectation = (final_probs * rewards).sum().item()
    assert final_expectation > 2.7
    return direct_gradient, score_gradient, final_probs, final_expectation


def main():
    torch.manual_seed(2026)
    torch.set_num_threads(1)
    initial, final, probs = fit_sft()
    print(f"SFT: {initial:.3f} -> {final:.3f}; P(0)={probs[0]:.3f}")
    initial, final, probs = fit_dpo()
    print(f"DPO: {initial:.3f} -> {final:.3f}; P(chosen)={probs[1]:.3f}")
    direct, score, probs, expectation = fit_expected_reward()
    print(f"RL: exact and score gradients {direct.tolist()} {score.tolist()}; "
          f"E[R]={expectation:.3f}")
    print("PASS: synthetic two-action objective tests; no claims about LLM generalization")


if __name__ == "__main__":
    main()
