"""Attentive-pooling probe, matching how V-JEPA actually evaluates frozen
features (src/models/attentive_pooler.py in facebookresearch/jepa): a
single learnable query token cross-attends over ALL of the encoder's
output tokens, instead of mean-pooling them uniformly. The query can learn
to weight whichever tokens matter for the downstream target -- cheap to
test against an already-trained checkpoint, no retraining of the backbone
needed. See docs/phase2-plan.md for why this is worth checking: mean-
pooling may be diluting whatever agent-position signal exists in a token
local to the agent.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.dataset import ClipDataset
from model.jepa import JEPA, JepaConfig


class AttentivePooler(nn.Module):
    """One learnable query cross-attends over the full token sequence,
    followed by a small residual MLP -- same shape as V-JEPA's
    AttentivePooler (query + CrossAttentionBlock), collapsed to a single
    pooled vector per clip."""

    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm_out = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, D) -> (B, D)"""
        b = x.shape[0]
        q = self.norm_q(self.query.expand(b, -1, -1))
        kv = self.norm_kv(x)
        out, _ = self.attn(q, kv, kv, need_weights=False)
        out = out + self.mlp(self.norm_out(out))
        return out.squeeze(1)


class AttentiveRegressor(nn.Module):
    def __init__(self, dim: int, out_dim: int = 2, heads: int = 4):
        super().__init__()
        self.pooler = AttentivePooler(dim, heads=heads)
        self.head = nn.Linear(dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pooler(x))


@torch.no_grad()
def embed_all_tokens(model, loader, device):
    """Full per-token embeddings (not pooled) -- needed since the whole
    point of the attentive pooler is to learn which tokens matter."""
    feats, targets = [], []
    for clip, agent_position in loader:
        clip = clip.to(device)
        tokens = model.patch_norm(model.patch_embed(clip)) + model.pos_embed
        out = model.context_encoder(tokens)  # (B, N, D), full clip, no masking
        feats.append(out.cpu())
        targets.append(agent_position)
    return torch.cat(feats, dim=0), torch.cat(targets, dim=0)


def train_and_eval_attentive_probe(
    feats: torch.Tensor, targets: torch.Tensor, device, val_frac: float = 0.2,
    steps: int = 500, batch_size: int = 64, lr: float = 1e-3,
):
    n, _, dim = feats.shape
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    n_val = int(n * val_frac)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    feats_train = feats[train_idx].to(device)
    targets_train = targets[train_idx].to(device)
    feats_val = feats[val_idx].to(device)
    targets_val = targets[val_idx].to(device)

    probe = AttentiveRegressor(dim).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)

    n_train = feats_train.shape[0]
    for step in range(steps):
        idx = torch.randint(0, n_train, (batch_size,), device=device)
        pred = probe(feats_train[idx])
        loss = nn.functional.mse_loss(pred, targets_train[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()

    probe.eval()
    with torch.no_grad():
        pred_val = probe(feats_val)
        mse = nn.functional.mse_loss(pred_val, targets_val).item()
    return mse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="dataset/pilot")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--clip-len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--probe-steps", type=int, default=500)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = Path(__file__).resolve().parent.parent / args.data_dir

    val_ds = ClipDataset(
        data_dir, clip_len=args.clip_len, stride=args.stride, split="val", shuffle=False
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, drop_last=True, num_workers=2)
    print(f"val clips: {len(val_ds)}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = JepaConfig(**ckpt["cfg"])

    trained = JEPA(cfg).to(device)
    trained.load_state_dict(ckpt["model"])
    trained.eval()

    random_init = JEPA(cfg).to(device)
    random_init.eval()

    for name, model in [("trained", trained), ("random_init", random_init)]:
        feats, targets = embed_all_tokens(model, val_loader, device)
        mse = train_and_eval_attentive_probe(feats, targets, device, steps=args.probe_steps)
        print(f"{name}: attentive-probe MSE on agent_position = {mse:.4f} (tokens {feats.shape[1]}, dim {feats.shape[2]})")


if __name__ == "__main__":
    main()
