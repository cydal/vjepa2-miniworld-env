"""Generate the pilot corpus (500-1000 episodes by default) and a
manifest.jsonl indexing them by layout_id/appearance_id for later splitting
(section 11 -- splits are by config, never by frame).

Run with:
    xvfb-run -a <conda env>/bin/python scripts/generate_pilot.py --n-episodes 500
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_gen.episode_generator import generate_episode
from data_gen.storage import write_episode
from env_wrapper.config import WrapperConfig
from env_wrapper.wrapper import MiniWorldJepaEnv


def disk_free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-episodes", type=int, default=500)
    parser.add_argument("--episode-len", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="dataset/pilot")
    parser.add_argument("--min-free-gb", type=float, default=2.0)
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent.parent / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    config = WrapperConfig()
    config.episode.max_steps = args.episode_len
    env = MiniWorldJepaEnv(config=config, render_mode="rgb_array")

    manifest_path = out_dir / "manifest.jsonl"
    t_start = time.time()
    with open(manifest_path, "w") as manifest_f:
        for i in range(args.n_episodes):
            seed = args.seed_start + i
            episode_id = f"ep{seed:06d}"
            episode = generate_episode(
                env, episode_id, seed=seed, momentum=config.episode.momentum
            )
            row = write_episode(episode, out_dir)
            manifest_f.write(json.dumps(row) + "\n")

            free_gb = disk_free_gb(out_dir)
            if free_gb < args.min_free_gb:
                print(
                    f"stopping early at episode {i}: only {free_gb:.2f} GB free "
                    f"(min {args.min_free_gb} GB)"
                )
                break

            if (i + 1) % 25 == 0 or i == 0:
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed
                print(
                    f"[{i + 1}/{args.n_episodes}] {elapsed:.1f}s elapsed, "
                    f"{rate:.2f} ep/s, {free_gb:.2f} GB free"
                )

    env.close()
    elapsed = time.time() - t_start
    du = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file())
    print(f"done: {elapsed:.1f}s total, dataset size {du / 1e6:.1f} MB, wrote to {out_dir}")


if __name__ == "__main__":
    main()
