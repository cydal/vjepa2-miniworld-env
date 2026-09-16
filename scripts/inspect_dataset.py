"""Pilot checklist from the brief: verify generation, inspect diversity,
estimate storage/time -- without building any model/probe code yet.

Run with (no display needed, this only reads files):
    <conda env>/bin/python scripts/inspect_dataset.py --dataset-dir dataset/pilot
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import imageio.v2 as imageio
import numpy as np
from PIL import Image


def load_manifest(dataset_dir: Path):
    rows = []
    with open(dataset_dir / "manifest.jsonl") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def contact_sheet(dataset_dir: Path, rows: list, out_path: Path, grid=(5, 5)):
    """Sample rows spread across distinct appearance_ids so the sheet
    actually shows 'same kind of room, different look' (section 8)."""
    by_appearance = {}
    for r in rows:
        by_appearance.setdefault(r["appearance_id"], r)
    sampled = list(by_appearance.values())[: grid[0] * grid[1]]

    thumbs = []
    for r in sampled:
        video_path = dataset_dir / r["path"] / "frames.mp4"
        reader = imageio.get_reader(video_path)
        mid_frame = reader.get_data(reader.count_frames() // 2)
        thumbs.append(Image.fromarray(mid_frame).resize((96, 96)))

    cols, rows_n = grid
    sheet = Image.new("RGB", (cols * 96, rows_n * 96), color=(20, 20, 20))
    for idx, thumb in enumerate(thumbs):
        x, y = (idx % cols) * 96, (idx // cols) * 96
        sheet.paste(thumb, (x, y))
    sheet.save(out_path)
    return len(thumbs)


FORWARD_ACTION = 2


def dynamics_stats(dataset_dir: Path, rows: list) -> dict:
    """Counts specific to why this environment replaced the MiniWorld
    OneRoom pilot (docs/phase1-plan.md): not just texture/appearance
    variety, but whether the *dynamics* we built this for actually landed
    in the recorded data across the corpus, not just in a hand-picked demo
    run."""
    door_opened = key_picked_up = obstacle_moved = goal_reached = blocked_move = 0
    total_forward_attempts = 0

    for r in rows:
        meta = json.load(open(dataset_dir / r["path"] / "meta.json"))
        steps = meta["steps"]

        if any(s["door_is_open"] for s in steps):
            door_opened += 1
        if any(s["carrying"] and s["carrying"]["type"] == "key" for s in steps):
            key_picked_up += 1
        if any(
            steps[i]["obstacle_positions"] != steps[i - 1]["obstacle_positions"]
            for i in range(1, len(steps))
        ):
            obstacle_moved += 1
        if steps[-1]["agent_position"] == steps[-1]["goal_position"]:
            goal_reached += 1

        episode_blocked = False
        for i in range(1, len(steps)):
            if steps[i]["action"] == FORWARD_ACTION:
                total_forward_attempts += 1
                if steps[i]["agent_position"] == steps[i - 1]["agent_position"]:
                    episode_blocked = True
        if episode_blocked:
            blocked_move += 1

    n = len(rows)
    return {
        "door_opened": (door_opened, n),
        "key_picked_up": (key_picked_up, n),
        "obstacle_moved_independently": (obstacle_moved, n),
        "goal_reached": (goal_reached, n),
        "had_a_blocked_move": (blocked_move, n),
        "total_forward_attempts_sampled": total_forward_attempts,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, default="dataset/pilot")
    args = parser.parse_args()

    dataset_dir = Path(__file__).resolve().parent.parent / args.dataset_dir
    rows = load_manifest(dataset_dir)

    n_episodes = len(rows)
    n_frames = sum(r["n_frames"] for r in rows)
    n_layouts = len(set(r["layout_id"] for r in rows))
    n_appearances = len(set(r["appearance_id"] for r in rows))
    disk_bytes = sum(f.stat().st_size for f in dataset_dir.rglob("*") if f.is_file())

    lengths = Counter(r["n_frames"] for r in rows)

    action_counts = Counter()
    for r in rows[: min(50, len(rows))]:  # sampling is enough for a sanity histogram
        meta = json.load(open(dataset_dir / r["path"] / "meta.json"))
        for step in meta["steps"]:
            if step["action"] is not None:
                action_counts[step["action"]] += 1

    print(f"episodes: {n_episodes}")
    print(f"total frames: {n_frames}")
    print(f"distinct layout_ids: {n_layouts}")
    print(f"distinct appearance_ids: {n_appearances}")
    print(f"disk usage: {disk_bytes / 1e6:.1f} MB ({disk_bytes / max(n_episodes, 1) / 1e3:.1f} KB/episode)")
    print(f"episode length distribution (top 5): {lengths.most_common(5)}")
    print(
        "action histogram (0=left,1=right,2=fwd,3=pickup,4=drop,5=toggle), "
        f"sampled: {dict(action_counts)}"
    )

    print("\ndynamics check (this is what this iteration was actually for):")
    for key, value in dynamics_stats(dataset_dir, rows).items():
        if isinstance(value, tuple):
            count, n = value
            print(f"  {key}: {count}/{n} episodes ({100 * count / n:.0f}%)")
        else:
            print(f"  {key}: {value}")

    out_path = dataset_dir / "contact_sheet.png"
    n_shown = contact_sheet(dataset_dir, rows, out_path)
    print(f"wrote contact sheet with {n_shown} distinct-appearance samples to {out_path}")


if __name__ == "__main__":
    main()
