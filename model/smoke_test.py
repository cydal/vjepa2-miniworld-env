"""Fast architecture check: one forward/backward pass on a random batch,
no data needed. Confirms loss is finite, context-encoder gets nonzero
gradients, and an EMA step actually moves target-encoder params -- catches
shape/wiring bugs before spending GPU time on a real training run.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from model.jepa import JEPA, JepaConfig


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = JepaConfig()
    model = JEPA(cfg).to(device)

    batch = torch.rand(4, cfg.clip_len, cfg.in_channels, cfg.img_size, cfg.img_size, device=device)

    target_before = model.target_encoder.blocks[0].mlp[0].weight.clone()

    loss, stats = model(batch)
    assert torch.isfinite(loss), f"loss is not finite: {loss}"
    print(f"loss: {loss.item():.4f}, stats: {stats}")

    loss.backward()
    ctx_grad_norm = sum(
        p.grad.norm().item() for p in model.context_encoder.parameters() if p.grad is not None
    )
    assert ctx_grad_norm > 0, "context encoder got zero gradient"
    print(f"context encoder grad norm: {ctx_grad_norm:.4f}")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    opt.step()
    model.update_target_encoder()
    target_after = model.target_encoder.blocks[0].mlp[0].weight
    moved = (target_before - target_after).abs().sum().item()
    assert moved > 0, "EMA update did not move target encoder params"
    print(f"target encoder moved by (sum abs diff): {moved:.6f}")

    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
