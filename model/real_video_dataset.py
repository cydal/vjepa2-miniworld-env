"""Real-video clip dataset for the low-cost Kinetics-mini demo (see
scripts/download_kinetics_mini.sh). Same (T, C, H, W) float-in-[0,1] clip
convention as model.dataset.ClipDataset, but reads plain mp4 files from
dataset/kinetics_mini/{train,val}/<class>/*.mp4 directly -- no
manifest.jsonl/meta.json, no position labels (not needed: this dataset only
feeds decoder/predictor reconstruction, not the position probe). Kinetics
frames aren't 128x128 like our car-nav renders, so each clip is resized
after decoding.
"""
import random
from pathlib import Path
from typing import Iterator, Tuple

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset

from data_gen.clip_extractor import extract_clips


class RealVideoClipDataset(IterableDataset):
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        clip_len: int = 16,
        stride: int = 8,
        img_size: int = 128,
        frame_stride: int = 2,
        shuffle: bool = True,
    ):
        self.paths = sorted(Path(data_dir, split).rglob("*.mp4"))
        self.clip_len = clip_len
        self.stride = stride
        self.img_size = img_size
        self.frame_stride = frame_stride
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        paths = list(self.paths)
        if self.shuffle:
            random.shuffle(paths)

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            paths = paths[worker_info.id :: worker_info.num_workers]

        for path in paths:
            frames = np.stack(imageio.mimread(path, memtest=False), axis=0)[:: self.frame_stride]  # (N, H, W, 3)
            clips = extract_clips(frames, clip_len=self.clip_len, stride=self.stride)
            if self.shuffle:
                random.shuffle(clips)
            for _, clip in clips:
                clip_t = torch.from_numpy(clip).float() / 255.0
                clip_t = clip_t.permute(0, 3, 1, 2)  # (T, C, H, W)
                clip_t = F.interpolate(
                    clip_t, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False
                )
                yield clip_t, torch.zeros(1)  # dummy label, unused
