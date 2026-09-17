"""Dataset for Phase 3 (action-conditioned post-training): same streaming/
chunked design as model.dataset.ClipDataset (bounded memory regardless of
corpus size -- see that module's docstring for why), but also assembles
one (action, state) pair per temporal tubelet, not just a single
end-of-clip position label.

No new data collection needed: every field used here (action, speed,
steer_angle, heading) is already recorded per-step in meta.json by
data_gen/car_episode_generator.py. This is purely a re-indexing job --
picking the tubelet-aligned frame for each of the clip's T tubelets.
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
from model.ac_predictor import ACTION_DIM, STATE_DIM
from model.dataset import _load_manifest_rows


def _action_state_at(step: dict) -> Tuple[np.ndarray, np.ndarray]:
    action = step.get("action")
    action = np.zeros(ACTION_DIM, dtype=np.float32) if action is None else np.asarray(action, dtype=np.float32)
    state = np.array([step["speed"], step["steer_angle"], step["heading"]], dtype=np.float32)
    assert action.shape == (ACTION_DIM,) and state.shape == (STATE_DIM,)
    return action, state


class ACClipDataset(IterableDataset):
    def __init__(
        self,
        data_dir: str,
        clip_len: int = 16,
        tubelet_size: int = 2,
        stride: int = 8,
        split: str = "train",
        val_every: int = 10,
        chunk_size: int = 32,
        shuffle: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.clip_len = clip_len
        self.tubelet_size = tubelet_size
        self.stride = stride
        self.chunk_size = chunk_size
        self.shuffle = shuffle
        self.rows = _load_manifest_rows(self.data_dir, split, val_every)
        self._n_clips = sum(
            max(0, (row["n_frames"] - clip_len) // stride + 1) for row in self.rows
        )

    def __len__(self) -> int:
        return self._n_clips

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        rows = list(self.rows)
        if self.shuffle:
            random.shuffle(rows)

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            rows = rows[worker_info.id :: worker_info.num_workers]

        for chunk_start in range(0, len(rows), self.chunk_size):
            chunk = rows[chunk_start : chunk_start + self.chunk_size]
            items = []
            for row in chunk:
                ep_dir = self.data_dir / row["path"]
                frames = np.stack(imageio.mimread(ep_dir / "frames.mp4"), axis=0)
                meta = json.loads((ep_dir / "meta.json").read_text())
                for start, _ in extract_clips(frames, clip_len=self.clip_len, stride=self.stride):
                    items.append((frames, meta, start))

            if self.shuffle:
                random.shuffle(items)

            for frames, meta, start in items:
                clip = frames[start : start + self.clip_len]
                clip_t = torch.from_numpy(clip).float() / 255.0
                clip_t = clip_t.permute(0, 3, 1, 2)  # (T_raw, C, H, W)

                n_tubelets = self.clip_len // self.tubelet_size
                actions, states = [], []
                for i in range(n_tubelets):
                    last_raw_idx = start + i * self.tubelet_size + self.tubelet_size - 1
                    a, s = _action_state_at(meta["steps"][last_raw_idx])
                    actions.append(a)
                    states.append(s)
                actions_t = torch.from_numpy(np.stack(actions))  # (n_tubelets, ACTION_DIM)
                states_t = torch.from_numpy(np.stack(states))    # (n_tubelets, STATE_DIM)

                yield clip_t, actions_t, states_t
