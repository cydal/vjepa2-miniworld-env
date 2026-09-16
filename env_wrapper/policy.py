"""Action policy used only to generate trajectories (section 4 of the
original brief) -- the JEPA model never sees these actions during
pretraining, they exist purely to produce natural-looking motion and are
kept in the episode metadata for a possible action-conditioned use later.

Pure random-walk movement rarely reaches the key, then the door, then the
goal within one episode's step budget -- an early version of this policy
(momentum-biased random walk only) exercised the door-open event in just
6/20 dry-run episodes. So this policy is layered:

1. **Reactive, absolute**: if the cell directly ahead is the key and the
   agent isn't carrying anything, pick it up; if the cell ahead is the
   locked door and the agent is carrying the matching key, toggle it.
2. **Greedy-with-noise, otherwise**: with probability `momentum`, turn/move
   toward whichever of {key, door, goal} is currently the live objective
   (a simple one-step-lookahead direction heuristic, not real pathfinding
   -- it can and does get stuck oscillating against a wall or a dynamic
   obstacle, which is fine, that's a real "blocked" transition, not a bug
   to route around); otherwise take a uniform-random movement action, so
   episodes aren't all identical shortest-paths.

Still fully scripted, no RL/learning involved.
"""
import numpy as np
from minigrid.core.world_object import Door, Key

TURN_LEFT, TURN_RIGHT, FORWARD, PICKUP, DROP, TOGGLE = 0, 1, 2, 3, 4, 5
MOVE_ACTIONS = (TURN_LEFT, TURN_RIGHT, FORWARD)


def _current_target(env):
    if env.carrying is None:
        return env.key.cur_pos
    if env.door.is_locked:
        return env.door.cur_pos
    return env.goal_pos


def _greedy_move(env, front_cell, rng: np.random.Generator) -> int:
    target = _current_target(env)
    dx, dy = target[0] - env.agent_pos[0], target[1] - env.agent_pos[1]

    if dx == 0 and dy == 0:
        desired_dir = env.agent_dir
    elif abs(dx) >= abs(dy):
        desired_dir = 0 if dx > 0 else 2
    else:
        desired_dir = 1 if dy > 0 else 3

    if desired_dir == env.agent_dir:
        blocked = front_cell is not None and not front_cell.can_overlap()
        if blocked:
            return int(rng.choice((TURN_LEFT, TURN_RIGHT)))
        return FORWARD

    diff = (desired_dir - env.agent_dir) % 4
    return TURN_RIGHT if diff in (1, 2) else TURN_LEFT


class MomentumRandomPolicy:
    def __init__(self, momentum: float, rng: np.random.Generator):
        self.momentum = momentum
        self.rng = rng

    def reset(self):
        pass

    def act(self, env) -> int:
        # `env` may be wrapped (see env_wrapper.wrapper.make_env); gymnasium
        # wrappers don't forward arbitrary attribute access, so reach
        # through to the raw MiniGridJepaEnv explicitly.
        env = env.unwrapped
        front_cell = env.grid.get(*env.front_pos)

        if isinstance(front_cell, Key) and env.carrying is None:
            return PICKUP
        if (
            isinstance(front_cell, Door)
            and front_cell.is_locked
            and isinstance(env.carrying, Key)
            and env.carrying.color == front_cell.color
        ):
            return TOGGLE

        if self.rng.random() < self.momentum:
            return _greedy_move(env, front_cell, self.rng)
        return int(self.rng.choice(MOVE_ACTIONS))
