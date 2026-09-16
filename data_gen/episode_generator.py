"""One episode = randomize (via env.reset(seed=...)) -> roll out the
momentum policy -> record every frame + per-step metadata (section 9).
"""
from dataclasses import dataclass, field
from typing import List

import numpy as np

from env_wrapper.policy import MomentumRandomPolicy
from env_wrapper.wrapper import MiniWorldJepaEnv


@dataclass
class Episode:
    episode_id: str
    environment_id: str
    layout: dict
    appearance: dict
    frames: np.ndarray  # (T, H, W, 3) uint8
    steps: List[dict] = field(default_factory=list)  # per-timestep metadata


def generate_episode(
    env: MiniWorldJepaEnv, episode_id: str, seed: int, momentum: float
) -> Episode:
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    policy = MomentumRandomPolicy(momentum=momentum, rng=rng)

    frames = [obs]
    steps = [{"timestep": 0, "action": None, **env.state_metadata()}]

    max_steps = env.jepa_config.episode.max_steps
    for t in range(1, max_steps + 1):
        action = policy.act()
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(obs)
        steps.append({"timestep": t, "action": int(action), **env.state_metadata()})
        if terminated or truncated:
            break

    return Episode(
        episode_id=episode_id,
        environment_id="oneroom",
        layout=dict(env.last_layout),
        appearance=dict(env.last_appearance),
        frames=np.stack(frames, axis=0),
        steps=steps,
    )
