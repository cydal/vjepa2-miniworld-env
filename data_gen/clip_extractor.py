"""Overlapping fixed-length clip extraction from one episode's frames
(section 6 of the brief): many training clips from a single simulator
rollout, no extra simulation needed.
"""
from typing import List, Tuple

import numpy as np


def extract_clips(
    frames: np.ndarray, clip_len: int = 32, stride: int = 8
) -> List[Tuple[int, np.ndarray]]:
    """Returns a list of (start_index, clip) pairs. Episodes shorter than
    clip_len yield no clips -- callers should generate episodes longer than
    the clip length they intend to train on."""
    n = frames.shape[0]
    clips = []
    for start in range(0, n - clip_len + 1, stride):
        clips.append((start, frames[start : start + clip_len]))
    return clips
