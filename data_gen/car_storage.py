"""Write one Car-Navigation-Env episode to disk, same layout as
data_gen/storage.py (frames.mp4 + meta.json + manifest row) so
model/dataset.py's ClipDataset reads it unchanged."""
import json
from pathlib import Path

import imageio.v2 as imageio

from data_gen.car_episode_generator import CarEpisode


def write_car_episode(episode: CarEpisode, out_dir: Path) -> dict:
    ep_dir = out_dir / episode.episode_id
    ep_dir.mkdir(parents=True, exist_ok=True)

    video_path = ep_dir / "frames.mp4"
    # crf=18 (default is much higher/lossier) -- file size difference is
    # negligible at this frame count (measured ~60KB -> ~100KB/episode) but
    # measurably closer to the live-rendered frames (MAE ~1.8 vs ~2.1/255).
    # The bigger gap vs. "watching it live" is resolution, not compression:
    # this writes at image_size=128 (matching the model's 8x8 patch grid),
    # while an interactive demo window renders far larger (~960x540).
    imageio.mimwrite(video_path, list(episode.frames), fps=10, codec="libx264",
                      output_params=["-crf", "18"])

    meta = {
        "episode_id": episode.episode_id,
        "environment_id": "carnav_gapfollower",
        "n_frames": int(episode.frames.shape[0]),
        "steps": episode.steps,
    }
    with open(ep_dir / "meta.json", "w") as f:
        json.dump(meta, f)

    last = episode.steps[-1]
    return {
        "episode_id": episode.episode_id,
        "environment_id": "carnav_gapfollower",
        "n_frames": meta["n_frames"],
        "path": str(ep_dir.relative_to(out_dir)),
        "reason": last.get("reason"),
        "is_success": last.get("is_success"),
        "targets_reached": last.get("targets_reached"),
    }
