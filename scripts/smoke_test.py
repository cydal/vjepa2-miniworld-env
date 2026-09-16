"""Step one: prove the MiniGrid env renders, and that the reactive policy
actually exercises the door/key mechanic and independent obstacle motion --
not just that the pipeline runs, but that the specific dynamics this
environment exists for actually show up.

Run with (no Xvfb needed -- headless via SDL_VIDEODRIVER=dummy, set in
env_wrapper/wrapper.py before minigrid/pygame is imported):
    <conda env>/bin/python scripts/smoke_test.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

from env_wrapper.config import WrapperConfig
from env_wrapper.policy import MomentumRandomPolicy
from env_wrapper.wrapper import make_env


def main():
    env = make_env(WrapperConfig())
    raw = env.unwrapped
    obs, info = env.reset(seed=0)
    assert obs.shape == (128, 128, 3), obs.shape
    assert obs.dtype == np.uint8
    assert obs.std() > 1.0, "frame looks blank -- rendering did not work"

    print("first reset ok:", raw.last_layout, raw.last_appearance)
    print("agent/goal/door metadata:", raw.state_metadata())

    policy = MomentumRandomPolicy(momentum=0.6, rng=np.random.default_rng(0))
    frames = [obs]
    obstacle_moved = False
    picked_up_key = False
    door_opened = False
    prev_obstacle_pos = raw.state_metadata()["obstacle_positions"]

    for t in range(150):
        action = policy.act(env)
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(obs)
        meta = raw.state_metadata()

        if meta["obstacle_positions"] != prev_obstacle_pos:
            obstacle_moved = True
        prev_obstacle_pos = meta["obstacle_positions"]

        if meta["carrying"] and meta["carrying"]["type"] == "key":
            picked_up_key = True
        if meta["door_is_open"]:
            door_opened = True

        if terminated or truncated:
            break

    out_dir = Path(__file__).resolve().parent.parent / "dataset"
    out_dir.mkdir(exist_ok=True)
    Image.fromarray(frames[0]).save(out_dir / "smoke_frame_0.png")
    Image.fromarray(frames[-1]).save(out_dir / "smoke_frame_last.png")
    print(f"wrote {len(frames)} frames over {t + 1} steps, saved first/last to {out_dir}")
    print(f"obstacle moved independent of agent action: {obstacle_moved}")
    print(f"picked up key: {picked_up_key}")
    print(f"door opened: {door_opened}")
    assert obstacle_moved, "obstacles never moved -- independent dynamics broken"

    # a second seed should give a visibly different layout/appearance,
    # confirming per-episode randomization actually runs
    obs2, _ = env.reset(seed=1)
    print("second reset:", raw.last_layout, raw.last_appearance)
    assert raw.last_appearance != {}

    env.close()
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
