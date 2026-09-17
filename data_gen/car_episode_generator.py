"""Episode generation for Car-Navigation-Env (3D car-nav, real independent
world dynamics via rule-based traffic + traffic lights, continuous
throttle/brake/steer actions). Written to the exact same manifest/mp4/
meta.json schema as data_gen/{episode_generator,storage}.py so the rest of
the pipeline (model/dataset.py's ClipDataset, train.py, finetune.py,
probe.py) works against it unchanged -- each per-step dict includes an
`agent_position` key ([x, y]) matching what ClipDataset expects.

The GapFollower scripted baseline (Car-Navigation-Env/baselines/scripted.py)
is used as the driving policy instead of a random policy: a random policy
crashes into a building almost immediately (README-documented, mean reward
-103, 0% success), which would give a dataset of near-identical one-second
crash clips instead of real driving footage.
"""
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np

CARNAV_ROOT = Path("/home/ubuntu/world_models/Car-Navigation-Env")
sys.path.insert(0, str(CARNAV_ROOT))

import carnav
from baselines.scripted import GapFollower


@dataclass
class CarEpisode:
    episode_id: str
    frames: np.ndarray  # (T, H, W, 3) uint8
    steps: List[dict] = field(default_factory=list)


def make_env(image_size: int = 128, seed: int = 0):
    """image_size=128 to match the MiniGrid pilot's frame size (and our
    model's patch grid: 128/16=8x8 spatial tokens) rather than the
    library's 64x64 default.

    Traffic/signal density cranked up from the library defaults
    (n_traffic=8, max_signals=6) -- for JEPA-style pretraining we want as
    much independent world motion and as many visible state-change events
    (light phase changes) as the env can produce, not just enough for a
    learnable RL task."""
    return carnav.make(
        obs_type="both", seed=seed, image_size=image_size,
        n_traffic=24, n_parked=20, traffic_speed=11.0,
        max_signals=14, signal_min_sep=25.0,
    )


def generate_episode(env, episode_id: str, seed: int, max_steps: int = 300) -> CarEpisode:
    policy = GapFollower()
    obs, info = env.reset(seed=seed)
    policy.reset()

    frames = [obs["image"]]
    steps = [{
        "timestep": 0, "action": None, "agent_position": [info["x"], info["y"]],
        **info,
    }]

    for t in range(1, max_steps + 1):
        action = policy.act(obs["vector"])
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(obs["image"])
        steps.append({
            "timestep": t, "action": [float(a) for a in action],
            "agent_position": [info["x"], info["y"]], "reward": reward,
            **info,
        })
        if terminated or truncated:
            break

    return CarEpisode(
        episode_id=episode_id,
        frames=np.stack(frames, axis=0),
        steps=steps,
    )
