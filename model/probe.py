"""Linear probe: does the trained encoder's representation carry more
information about agent position than a random-init encoder of the same
architecture? A loss curve going down isn't enough evidence that scaling up
data generation is worth it -- this is the actual signal. See
docs/phase2-plan.md.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

from model.dataset import ClipDataset
from model.jepa import JEPA, JepaConfig


@torch.no_grad()
def embed_all(model, loader, device):
    feats, targets = [], []
    for clip, agent_position in loader:
        clip = clip.to(device)
        tokens = model.patch_embed(clip) + model.pos_embed
        out = model.context_encoder(tokens)  # full clip, no masking, for probing
        feats.append(out.mean(dim=1).cpu().numpy())
        targets.append(agent_position.numpy())
    return np.concatenate(feats, axis=0), np.concatenate(targets, axis=0)


def fit_and_eval_linear_probe(feats: np.ndarray, targets: np.ndarray, val_frac: float = 0.2):
    n = feats.shape[0]
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    n_val = int(n * val_frac)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    X_train = np.concatenate([feats[train_idx], np.ones((len(train_idx), 1))], axis=1)
    X_val = np.concatenate([feats[val_idx], np.ones((len(val_idx), 1))], axis=1)
    y_train, y_val = targets[train_idx], targets[val_idx]

    w, _, _, _ = np.linalg.lstsq(X_train, y_train, rcond=None)
    pred = X_val @ w
    mse = float(np.mean((pred - y_val) ** 2))
    return mse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="dataset/pilot")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--clip-len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
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
        feats, targets = embed_all(model, val_loader, device)
        mse = fit_and_eval_linear_probe(feats, targets)
        print(f"{name}: linear probe MSE on agent_position = {mse:.4f} (feat dim {feats.shape[1]})")


if __name__ == "__main__":
    main()
