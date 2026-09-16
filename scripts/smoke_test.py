"""Step one: prove Xvfb + pyglet + mesa softpipe can actually render a
MiniWorld frame on this box before anything else gets built on top of it.

Run with:
    xvfb-run -a <conda env>/bin/python scripts/smoke_test.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

from env_wrapper.config import WrapperConfig
from env_wrapper.policy import MomentumRandomPolicy
from env_wrapper.wrapper import MiniWorldJepaEnv


def main():
    env = MiniWorldJepaEnv(config=WrapperConfig(), render_mode="rgb_array")
    obs, info = env.reset(seed=0)
    assert obs.shape == (128, 128, 3), obs.shape
    assert obs.dtype == np.uint8
    assert obs.std() > 1.0, "frame looks blank -- rendering did not work"

    print("first reset ok:", env.last_layout, env.last_appearance)
    print("agent/goal metadata:", env.state_metadata())

    policy = MomentumRandomPolicy(momentum=0.6, rng=np.random.default_rng(0))
    frames = [obs]
    for _ in range(15):
        action = policy.act()
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(obs)
        if terminated or truncated:
            break

    out_dir = Path(__file__).resolve().parent.parent / "dataset"
    out_dir.mkdir(exist_ok=True)
    Image.fromarray(frames[0]).save(out_dir / "smoke_frame_0.png")
    Image.fromarray(frames[-1]).save(out_dir / "smoke_frame_last.png")
    print(f"wrote {len(frames)} frames, saved first/last to {out_dir}")

    # a second seed should give a visibly different room (different
    # appearance/geometry), confirming per-episode randomization actually runs
    obs2, _ = env.reset(seed=1)
    print("second reset:", env.last_layout, env.last_appearance)
    assert env.last_appearance != {}

    env.close()
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
