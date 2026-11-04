"""Tiny causal Transformer: zero pretrained weights; only synthetic data.

Requires PyTorch 2.x. This demonstrates tensor shapes, causal masking,
gradient descent and overfitting. It is not an LLM benchmark.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class CausalAttention(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.heads = heads
        self.head_dim = width // heads
        self.qkv = nn.Linear(width, 3 * width)
        self.project = nn.Linear(width, width)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def reshape(t):
            return t.view(batch, length, self.heads, self.head_dim).transpose(1, 2)

        q, k, v = (reshape(t) for t in (q, k, v))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        mask = torch.ones(length, length, dtype=torch.bool, device=x.device).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        context = weights @ v
        context = context.transpose(1, 2).contiguous().view(batch, length, width)
        return self.project(context)


class DecoderBlock(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attn = CausalAttention(width, heads)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, width * 4),
            nn.GELU(),
            nn.Linear(width * 4, width),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class TinyDecoder(nn.Module):
    def __init__(self, vocab=16, width=32, heads=4, layers=2, max_length=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab, width)
        self.position = nn.Embedding(max_length, width)
        self.blocks = nn.Sequential(*(DecoderBlock(width, heads) for _ in range(layers)))
        self.norm = nn.LayerNorm(width)
        self.logits = nn.Linear(width, vocab)

    def forward(self, tokens):
        batch, length = tokens.shape
        if length > self.position.num_embeddings:
            raise ValueError("sequence exceeds max_length")
        x = self.embedding(tokens)
        x = x + self.position(torch.arange(length, device=tokens.device))[None]
        return self.logits(self.norm(self.blocks(x)))


def run_checks():
    torch.set_num_threads(1)
    torch.manual_seed(2026)
    model = TinyDecoder()
    model.eval()

    # Same prefix, different suffix: logits for the prefix must be identical.
    a = torch.tensor([[1, 2, 3, 4, 5, 6]])
    b = torch.tensor([[1, 2, 3, 9, 8, 7]])
    with torch.no_grad():
        out_a, out_b = model(a), model(b)
    assert out_a.shape == (1, 6, 16)
    assert torch.allclose(out_a[:, :3], out_b[:, :3], atol=1e-6)
    assert not torch.allclose(out_a[:, 4:], out_b[:, 4:])

    # Small synthetic sequences. Targets are shifted by one token.
    sequences = torch.tensor([
        [1, 2, 3, 4, 5, 6, 7, 8],
        [1, 2, 3, 4, 5, 6, 7, 8],
        [1, 3, 5, 7, 9, 11, 13, 15],
        [1, 3, 5, 7, 9, 11, 13, 15],
        [1, 2, 4, 6, 8, 10, 12, 14],
        [1, 2, 4, 6, 8, 10, 12, 14],
    ])
    inputs, targets = sequences[:, :-1], sequences[:, 1:]
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.006, weight_decay=0)

    model.train()
    with torch.no_grad():
        initial = F.cross_entropy(model(inputs).reshape(-1, 16), targets.reshape(-1)).item()

    for _ in range(110):
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, 16), targets.reshape(-1))
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        final = F.cross_entropy(model(inputs).reshape(-1, 16), targets.reshape(-1)).item()
    assert final < initial * 0.5, (initial, final)
    print(f"PASS causal mask, shapes, gradients, synthetic loss: {initial:.3f} -> {final:.3f}")
    print("NOTE: memorizing repeated sequences does not demonstrate generalization.")


if __name__ == "__main__":
    run_checks()
