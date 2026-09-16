"""Clip dataset over a generated pilot corpus (dataset/pilot/ by default).
Decodes each episode's frames.mp4 once into RAM (the full 500-episode pilot
is ~1.9GB decoded -- small enough to skip a streaming loader entirely), then
slices overlapping clips via data_gen/clip_extractor.extract_clips.
"""
import json
from pathlib import Path
from typing import List, Tuple

import imageio.v2 as imageio
import numpy as np
import torch
from torch.utils.data import Dataset

from data_gen.clip_extractor import extract_clips


class ClipDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        clip_len: int = 16,
        stride: int = 8,
        split: str = "train",
        val_every: int = 10,
    ):
        data_dir = Path(data_dir)
        manifest = [
            json.loads(line)
            for line in (data_dir / "manifest.jsonl").read_text().splitlines()
        ]
        manifest.sort(key=lambda r: r["episode_id"])

        self.episodes = []  # list of (frames uint8 array, meta dict)
        self.index: List[Tuple[int, int]] = []  # (episode_idx, start)
        for i, row in enumerate(manifest):
            is_val = (i % val_every) == 0
            if (split == "val") != is_val:
                continue
            ep_dir = data_dir / row["path"]
            frames = imageio.mimread(ep_dir / "frames.mp4")
            frames = np.stack(frames, axis=0)  # (T, H, W, 3) uint8
            meta = json.loads((ep_dir / "meta.json").read_text())
            ep_idx = len(self.episodes)
            self.episodes.append((frames, meta))
            for start, _ in extract_clips(frames, clip_len=clip_len, stride=stride):
                self.index.append((ep_idx, start))

        self.clip_len = clip_len

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        ep_idx, start = self.index[idx]
        frames, meta = self.episodes[ep_idx]
        clip = frames[start : start + self.clip_len]  # (T, H, W, 3) uint8
        clip = torch.from_numpy(clip).float() / 255.0
        clip = clip.permute(0, 3, 1, 2)  # (T, C, H, W)
        last_step = meta["steps"][start + self.clip_len - 1]
        agent_position = torch.tensor(last_step["agent_position"], dtype=torch.float32)
        return clip, agent_position
