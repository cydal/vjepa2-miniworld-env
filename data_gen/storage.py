"""Write one episode to disk: frames as an mp4 (compact, keeps the pilot
inside the disk budget) plus a JSON metadata sidecar (section 9). The
per-step metadata -- position/orientation/goal/action -- is retained here
for future evaluation but is never part of what a training script reads as
model input; that's just `frames.mp4`.
"""
import json
from pathlib import Path

import imageio.v2 as imageio

from data_gen.episode_generator import Episode


def layout_id(layout: dict) -> str:
    return f"size{layout['size']:.1f}_decor{layout['n_decor']}"


def appearance_id(appearance: dict) -> str:
    return f"{appearance['wall_tex']}-{appearance['floor_tex']}-{appearance['ceil_tex']}"


def write_episode(episode: Episode, out_dir: Path) -> dict:
    """Writes out_dir/<episode_id>/{frames.mp4,meta.json} and returns the
    manifest row for this episode."""
    ep_dir = out_dir / episode.episode_id
    ep_dir.mkdir(parents=True, exist_ok=True)

    video_path = ep_dir / "frames.mp4"
    imageio.mimwrite(video_path, list(episode.frames), fps=10, codec="libx264")

    meta = {
        "episode_id": episode.episode_id,
        "environment_id": episode.environment_id,
        "layout": episode.layout,
        "appearance": episode.appearance,
        "n_frames": int(episode.frames.shape[0]),
        "steps": episode.steps,
    }
    with open(ep_dir / "meta.json", "w") as f:
        json.dump(meta, f)

    return {
        "episode_id": episode.episode_id,
        "environment_id": episode.environment_id,
        "layout_id": layout_id(episode.layout),
        "appearance_id": appearance_id(episode.appearance),
        "n_frames": meta["n_frames"],
        "path": str(ep_dir.relative_to(out_dir)),
    }
