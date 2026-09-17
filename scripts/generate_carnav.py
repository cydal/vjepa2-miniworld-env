"""Generate a Car-Navigation-Env episode corpus. Designed to be run as
several parallel OS processes (one per CPU core) rather than a Python
multiprocessing pool -- Panda3D's offscreen graphics context is not
guaranteed fork-safe, so each shard is a separate process from the start,
each writing its own manifest_<shard>.jsonl; merge_manifests.py combines
them after all shards finish.

Verified leak: RSS per process grows roughly linearly with reset() count
(~17MB/episode observed -- 1.67GB at 80 episodes, 4.85GB at 265, on this
15GB-RAM box) -- build_scene() rebuilds the whole procedural city mesh
every reset() (randomize_map=True), and something in that path (city
geometry, or GeomNode/RenderState caching under it) isn't fully released
by Panda3D's removeNode(). This caused two prior OOM-adjacent box crashes.
Rather than chase a Panda3D-internal leak, --episodes-per-process caps how
many episodes one process generates before exiting (letting the OS reclaim
everything), and use run_shard.sh below to loop fresh processes per batch,
appending to the same manifest_<shard>.jsonl across invocations.

Run (per shard, e.g. 4 shards on a 4-core box):
    <conda env>/bin/python scripts/generate_carnav.py \
        --n-episodes 625 --seed-start 0    --shard-id 0 --out-dir dataset/carnav_pilot
    <conda env>/bin/python scripts/generate_carnav.py \
        --n-episodes 625 --seed-start 625  --shard-id 1 --out-dir dataset/carnav_pilot
    ... (seed-start offsets by --n-episodes per shard, so ranges never overlap)
    For a single shard's full run in memory-safe batches, see run_shard.sh.
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_gen.car_episode_generator import generate_episode, make_env
from data_gen.car_storage import write_car_episode


def disk_free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-episodes", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="dataset/carnav_pilot")
    parser.add_argument("--min-free-gb", type=float, default=2.0)
    parser.add_argument("--append", action="store_true",
                         help="append to manifest_<shard>.jsonl instead of overwriting -- "
                              "for looping fresh processes over a shard in small batches")
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent.parent / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(image_size=args.image_size, seed=args.seed_start)

    manifest_path = out_dir / f"manifest_{args.shard_id}.jsonl"
    t_start = time.time()
    with open(manifest_path, "a" if args.append else "w") as manifest_f:
        for i in range(args.n_episodes):
            seed = args.seed_start + i
            episode_id = f"ep{seed:06d}"
            episode = generate_episode(env, episode_id, seed=seed, max_steps=args.max_steps)
            row = write_car_episode(episode, out_dir)
            manifest_f.write(json.dumps(row) + "\n")
            manifest_f.flush()

            free_gb = disk_free_gb(out_dir)
            if free_gb < args.min_free_gb:
                print(f"[shard {args.shard_id}] stopping early at episode {i}: "
                      f"only {free_gb:.2f} GB free")
                break

            if (i + 1) % 25 == 0 or i == 0:
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed
                print(f"[shard {args.shard_id}] [{i + 1}/{args.n_episodes}] "
                      f"{elapsed:.1f}s elapsed, {rate:.2f} ep/s, {free_gb:.2f} GB free")

    elapsed = time.time() - t_start
    print(f"[shard {args.shard_id}] done: {elapsed:.1f}s total, wrote to {manifest_path}")


if __name__ == "__main__":
    main()
