"""Clip dataset over a generated pilot corpus (dataset/pilot/ by default).

Streams episodes in shuffled chunks rather than decoding the whole corpus
into RAM up front: this box has 15GB total RAM, and a several-thousand-
episode corpus decodes to tens of GB (500 episodes was ~1.9GB, fine to hold
fully in RAM; scaling up breaks that). Each chunk of `chunk_size` episodes
is decoded, its clips are extracted and shuffled together, then the
decoded frames are dropped before moving to the next chunk -- peak memory
stays bounded by chunk_size regardless of corpus size. This is a block
shuffle (shuffled within a chunk, not globally) rather than a full
dataset-wide shuffle, which is an acceptable trade for staying within RAM.
"""
import json
import random
from pathlib import Path
from typing import Iterator, Tuple

import imageio.v2 as imageio
import numpy as np
import torch
from torch.utils.data import IterableDataset

from data_gen.clip_extractor import extract_clips


def _load_manifest_rows(data_dir: Path, split: str, val_every: int):
    manifest = [
        json.loads(line)
        for line in (data_dir / "manifest.jsonl").read_text().splitlines()
    ]
    manifest.sort(key=lambda r: r["episode_id"])
    rows = []
    for i, row in enumerate(manifest):
        is_val = (i % val_every) == 0
        if (split == "val") == is_val:
            rows.append(row)
    return rows


class ClipDataset(IterableDataset):
    def __init__(
        self,
        data_dir: str,
        clip_len: int = 16,
        stride: int = 8,
        split: str = "train",
        val_every: int = 10,
        chunk_size: int = 32,
        shuffle: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.clip_len = clip_len
        self.stride = stride
        self.chunk_size = chunk_size
        self.shuffle = shuffle
        self.rows = _load_manifest_rows(self.data_dir, split, val_every)
        # exact clip count from each episode's recorded n_frames -- no
        # decoding needed just to know how many clips extract_clips will
        # produce.
        self._n_clips = sum(
            max(0, (row["n_frames"] - clip_len) // stride + 1) for row in self.rows
        )

    def __len__(self) -> int:
        return self._n_clips

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        rows = list(self.rows)
        if self.shuffle:
            # shuffle before sharding so each DataLoader worker still gets
            # a randomized subset, not a contiguous (and therefore
            # correlated, since manifest is sorted by episode_id) slice.
            random.shuffle(rows)

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # decoding video is the bottleneck at corpus sizes beyond the
            # pilot -- split episodes disjointly across DataLoader workers
            # so multiple CPU cores decode in parallel instead of one
            # worker doing (and every worker duplicating) all the work.
            rows = rows[worker_info.id :: worker_info.num_workers]

        for chunk_start in range(0, len(rows), self.chunk_size):
            chunk = rows[chunk_start : chunk_start + self.chunk_size]
            items = []  # (frames uint8 array, meta dict, clip start)
            for row in chunk:
                ep_dir = self.data_dir / row["path"]
                frames = np.stack(imageio.mimread(ep_dir / "frames.mp4"), axis=0)
                meta = json.loads((ep_dir / "meta.json").read_text())
                for start, _ in extract_clips(frames, clip_len=self.clip_len, stride=self.stride):
                    items.append((frames, meta, start))

            if self.shuffle:
                random.shuffle(items)

            for frames, meta, start in items:
                clip = frames[start : start + self.clip_len]  # (T, H, W, 3) uint8
                clip_t = torch.from_numpy(clip).float() / 255.0
                clip_t = clip_t.permute(0, 3, 1, 2)  # (T, C, H, W)
                last_step = meta["steps"][start + self.clip_len - 1]
                agent_position = torch.tensor(last_step["agent_position"], dtype=torch.float32)
                yield clip_t, agent_position
