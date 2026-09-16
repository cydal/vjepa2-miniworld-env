"""One episode = randomize (via env.reset(seed=...)) -> roll out the
momentum policy -> record every frame + per-step metadata (section 9).
"""
from dataclasses import dataclass, field
from typing import List

import numpy as np

from env_wrapper.policy import MomentumRandomPolicy


@dataclass
class Episode:
    episode_id: str
    environment_id: str
    layout: dict
    appearance: dict
    frames: np.ndarray  # (T, H, W, 3) uint8
    steps: List[dict] = field(default_factory=list)  # per-timestep metadata


def generate_episode(env, episode_id: str, seed: int, momentum: float) -> Episode:
    """`env` is whatever `env_wrapper.wrapper.make_env()` returns -- a
    gymnasium-wrapped env whose reset()/step() yield a bare RGB frame as
    `obs`. Gymnasium wrappers don't forward arbitrary attribute access, so
    `last_layout`/`last_appearance`/`state_metadata()`/`jepa_config` are
    read off `env.unwrapped` (the raw MiniGridJepaEnv) explicitly."""
    raw = env.unwrapped
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    policy = MomentumRandomPolicy(momentum=momentum, rng=rng)

    frames = [obs]
    steps = [{"timestep": 0, "action": None, **raw.state_metadata()}]

    max_steps = raw.jepa_config.episode.max_steps
    for t in range(1, max_steps + 1):
        action = policy.act(env)
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(obs)
        steps.append({"timestep": t, "action": int(action), **raw.state_metadata()})
        if terminated or truncated:
            break

    return Episode(
        episode_id=episode_id,
        environment_id="minigrid_doorkey_dynobs",
        layout=dict(raw.last_layout),
        appearance=dict(raw.last_appearance),
        frames=np.stack(frames, axis=0),
        steps=steps,
    )
