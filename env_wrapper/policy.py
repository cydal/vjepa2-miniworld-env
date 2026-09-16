"""Action policy used only to generate trajectories (section 4 of the
brief) -- the JEPA model never sees these actions during pretraining, they
exist purely to produce natural-looking motion and are kept in the episode
metadata for a possible Phase 2 (action-conditioned) use.
"""
import numpy as np

TURN_LEFT, TURN_RIGHT, MOVE_FORWARD = 0, 1, 2
ACTIONS = (TURN_LEFT, TURN_RIGHT, MOVE_FORWARD)


class MomentumRandomPolicy:
    """Picks a uniform-random action, but with `momentum` probability just
    repeats the previous one. Pure per-step uniform random turn/forward
    produces visually jittery trajectories; biasing toward continuing
    whatever it was already doing makes clips look like short walks
    instead of noise, with no RL/reward involved.
    """

    def __init__(self, momentum: float, rng: np.random.Generator):
        self.momentum = momentum
        self.rng = rng
        self.prev_action = None

    def reset(self):
        self.prev_action = None

    def act(self) -> int:
        if self.prev_action is not None and self.rng.random() < self.momentum:
            action = self.prev_action
        else:
            action = int(self.rng.choice(ACTIONS))
        self.prev_action = action
        return action
