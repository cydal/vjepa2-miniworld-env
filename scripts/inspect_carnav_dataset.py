"""Sanity-check a generated Car-Navigation-Env corpus: episode length
distribution, termination-reason breakdown, target-reach rate, a contact
sheet of sample frames. Mirrors scripts/inspect_dataset.py's role for the
MiniGrid pilot."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import imageio.v2 as imageio
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, required=True)
    args = parser.parse_args()

    dataset_dir = Path(__file__).resolve().parent.parent / args.dataset_dir
    manifest = [json.loads(l) for l in (dataset_dir / "manifest.jsonl").read_text().splitlines()]

    n_frames_total = sum(r["n_frames"] for r in manifest)
    reasons = Counter(r.get("reason") for r in manifest)
    targets_reached = [r.get("targets_reached", 0) for r in manifest]
    disk_bytes = sum(f.stat().st_size for f in dataset_dir.rglob("*") if f.is_file())

    print(f"episodes: {len(manifest)}")
    print(f"total frames: {n_frames_total}")
    print(f"disk usage: {disk_bytes / 1e6:.1f} MB ({disk_bytes / 1e3 / len(manifest):.1f} KB/episode)")
    lengths = sorted(Counter(r["n_frames"] for r in manifest).items(), key=lambda kv: -kv[1])
    print(f"episode length distribution (top 5): {lengths[:5]}")
    print(f"termination reasons: {dict(reasons)}")
    print(f"targets reached: mean={np.mean(targets_reached):.2f}, "
          f"full-route (3/3): {sum(1 for t in targets_reached if t >= 3)}/{len(manifest)}")

    # contact sheet: first + last frame of a handful of episodes
    n_samples = min(8, len(manifest))
    rng = np.random.default_rng(0)
    sample_rows = rng.choice(manifest, size=n_samples, replace=False)
    tiles = []
    for row in sample_rows:
        ep_dir = dataset_dir / row["path"]
        frames = imageio.mimread(ep_dir / "frames.mp4")
        tiles.append(np.concatenate([frames[0], frames[-1]], axis=1))
    sheet = np.concatenate(tiles, axis=0)
    Image.fromarray(sheet).save(dataset_dir / "contact_sheet.png")
    print(f"wrote contact sheet ({n_samples} episodes, first|last frame) to {dataset_dir / 'contact_sheet.png'}")


if __name__ == "__main__":
    main()
