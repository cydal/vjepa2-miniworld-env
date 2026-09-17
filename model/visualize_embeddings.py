"""Visual validation of a fine-tuned encoder, for the article/demo rather
than a probe number: (1) a 2D projection of embeddings colored by known
state variables, to see whether the representation space organizes along
axes it was never told about, and (2) nearest-neighbor grids -- pick a
query frame, show the frames the model thinks are closest.

Not part of the training pipeline; run manually once a checkpoint exists.
Deliberately not wired into any automatic overnight run.
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

from model.jepa import JepaConfig
from model.pretrained_jepa import PretrainedJEPA


def sample_clips(data_dir: Path, split: str, clip_len: int, stride: int,
                  n_samples: int, val_every: int = 10, seed: int = 0):
    """Self-contained sampler (not model.dataset.ClipDataset) -- pulls extra
    state fields (heading, speed, reason) alongside the frames, which the
    training dataset intentionally doesn't expose."""
    manifest = [json.loads(l) for l in (data_dir / "manifest.jsonl").read_text().splitlines()]
    manifest.sort(key=lambda r: r["episode_id"])
    rows = [r for i, r in enumerate(manifest) if ((i % val_every) == 0) == (split == "val")]

    rng = random.Random(seed)
    rng.shuffle(rows)

    clips, metas = [], []
    for row in rows:
        if len(clips) >= n_samples:
            break
        ep_dir = data_dir / row["path"]
        meta = json.loads((ep_dir / "meta.json").read_text())
        n = meta["n_frames"]
        if n < clip_len:
            continue
        start = rng.randint(0, n - clip_len)
        frames = imageio.mimread(ep_dir / "frames.mp4")
        clip = np.stack(frames[start:start + clip_len], axis=0)
        last = meta["steps"][start + clip_len - 1]
        clips.append(clip)
        metas.append({
            "x": last.get("x"), "y": last.get("y"), "heading": last.get("heading"),
            "speed": last.get("speed"), "reason": last.get("reason"),
            "episode_id": row["episode_id"], "last_frame": clip[-1],
        })
    return clips, metas


@torch.no_grad()
def embed_clips(encoder, clips, device, batch_size=16):
    """Mean-pooled embedding per clip -- simplest pooling, fine for
    visualization (the attentive pooler is for the actual probe number)."""
    embs = []
    clips_t = torch.from_numpy(np.stack(clips)).float() / 255.0  # (N, T, H, W, C)
    clips_t = clips_t.permute(0, 1, 4, 2, 3)  # (N, T, C, H, W)
    for i in range(0, len(clips), batch_size):
        batch = clips_t[i:i + batch_size].to(device).permute(0, 2, 1, 3, 4)  # (B,C,T,H,W)
        out = encoder(batch)  # (B, N_tokens, D)
        embs.append(out.mean(dim=1).cpu().numpy())
    return np.concatenate(embs, axis=0)


def plot_embedding_space(emb2d, metas, out_path: Path):
    fields = [
        ("heading", "hsv", "Heading (rad)"),
        ("speed", "viridis", "Speed (m/s)"),
        ("x", "coolwarm", "X position"),
    ]
    fig, axes = plt.subplots(1, len(fields), figsize=(6 * len(fields), 5))
    for ax, (key, cmap, label) in zip(axes, fields):
        vals = np.array([m[key] for m in metas], dtype=float)
        sc = ax.scatter(emb2d[:, 0], emb2d[:, 1], c=vals, cmap=cmap, s=14, alpha=0.85)
        ax.set_title(label)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(sc, ax=ax, fraction=0.046)
    crashed = np.array([m["reason"] == "crash" for m in metas])
    fig.suptitle(f"Embedding space (t-SNE), n={len(metas)}, {crashed.sum()} crash-terminated clips marked separately")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"wrote {out_path}")


def plot_nearest_neighbors(embeddings, metas, out_path: Path, n_queries=4, k=5, seed=0):
    rng = np.random.default_rng(seed)
    query_idx = rng.choice(len(metas), size=n_queries, replace=False)
    dists = np.linalg.norm(embeddings[:, None, :] - embeddings[None, :, :], axis=-1)

    fig, axes = plt.subplots(n_queries, k + 1, figsize=((k + 1) * 2, n_queries * 2))
    for row, qi in enumerate(query_idx):
        order = np.argsort(dists[qi])
        neighbors = [j for j in order if j != qi][:k]
        axes[row, 0].imshow(metas[qi]["last_frame"])
        axes[row, 0].set_title("query", fontsize=9)
        axes[row, 0].axis("off")
        for col, ni in enumerate(neighbors, start=1):
            axes[row, col].imshow(metas[ni]["last_frame"])
            axes[row, col].set_title(f"d={dists[qi, ni]:.2f}", fontsize=8)
            axes[row, col].axis("off")
    fig.suptitle("Nearest neighbors in embedding space (query | top-5 closest)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--clip-len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--n-samples", type=int, default=300)
    parser.add_argument("--out-dir", type=str, default="notes/figures")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent.parent
    data_dir = repo / args.data_dir
    out_dir = repo / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = JepaConfig()
    model = PretrainedJEPA(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    clips, metas = sample_clips(data_dir, "val", args.clip_len, args.stride, args.n_samples)
    print(f"sampled {len(clips)} clips")

    embeddings = embed_clips(model.context_encoder, clips, device)
    emb2d = TSNE(n_components=2, init="pca", random_state=0).fit_transform(embeddings)

    plot_embedding_space(emb2d, metas, out_dir / "embedding_space.png")
    plot_nearest_neighbors(embeddings, metas, out_dir / "nearest_neighbors.png")


if __name__ == "__main__":
    main()
