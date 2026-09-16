"""Thin wrapper around MiniWorld's OneRoom: randomizes geometry, appearance,
and clutter per episode, while keeping the camera fixed.

Design note: MiniWorld's own `domain_rand=True` flag also randomizes camera
height/pitch/fov/forward-displacement (see `Agent.randomize` in
miniworld/entity.py, driven by miniworld/params.py's cam_* entries), which
we don't want -- the brief is explicit that variation should come from the
world, not from simultaneously changing the camera. So `_build_params()`
below takes a copy of MiniWorld's DEFAULT_PARAMS and pins every cam_* /
motion entry to its default while leaving sky_color / light_* /
obj_color_bias free to randomize. That gets us real lighting diversity
(section 7D of the brief) for free instead of writing our own lighting
code.
"""
from gymnasium import spaces, utils
from miniworld.entity import Ball, Box
from miniworld.miniworld import MiniWorldEnv
from miniworld.params import DEFAULT_PARAMS

from env_wrapper.config import WrapperConfig

_FIXED_PARAM_NAMES = (
    "forward_step",
    "forward_drift",
    "turn_step",
    "bot_radius",
    "cam_pitch",
    "cam_fov_y",
    "cam_height",
    "cam_fwd_disp",
)


def _build_params():
    params = DEFAULT_PARAMS.copy()
    pinned = DEFAULT_PARAMS.no_random()
    for name in _FIXED_PARAM_NAMES:
        params.params[name] = pinned.params[name]
    return params


class MiniWorldJepaEnv(MiniWorldEnv, utils.EzPickle):
    """OneRoom-style single room with per-episode geometry + appearance
    randomization.

    A `MiniWorldEnv` subclass rather than a gymnasium `Wrapper`, because the
    randomization has to happen inside `_gen_world`, which only has access
    to `self.np_random` seeded by that episode's `reset(seed=...)` -- there
    is no post-hoc wrapper hook that fires before world geometry is built.
    """

    def __init__(self, config: WrapperConfig = None, max_episode_steps=None, **kwargs):
        self.jepa_config = config or WrapperConfig()
        max_episode_steps = max_episode_steps or self.jepa_config.episode.max_steps
        obs_size = self.jepa_config.episode.obs_size

        # populated by _gen_world, read back by state_metadata()/generators
        self.last_layout: dict = {}
        self.last_appearance: dict = {}

        super().__init__(
            max_episode_steps=max_episode_steps,
            obs_width=obs_size,
            obs_height=obs_size,
            params=_build_params(),
            domain_rand=True,
            **kwargs,
        )
        utils.EzPickle.__init__(
            self, config=config, max_episode_steps=max_episode_steps, **kwargs
        )

        # movement-only action space: turn_left, turn_right, move_forward
        self.action_space = spaces.Discrete(self.actions.move_forward + 1)

    def _gen_world(self):
        geo = self.jepa_config.geometry
        app = self.jepa_config.appearance
        rng = self.np_random

        size = float(rng.uniform(geo.size_min, geo.size_max))
        wall_tex = str(rng.choice(app.wall_textures))
        floor_tex = str(rng.choice(app.floor_textures))
        ceil_tex = str(rng.choice(app.ceiling_textures))

        self.add_rect_room(
            min_x=0,
            max_x=size,
            min_z=0,
            max_z=size,
            wall_tex=wall_tex,
            floor_tex=floor_tex,
            ceil_tex=ceil_tex,
        )

        # the goal: a red box, visible but never given to the model as a label
        self.box = self.place_entity(Box(color="red"))

        # irrelevant clutter for object diversity (section 7E); never red,
        # so it can't be confused with the goal by anyone inspecting frames
        n_decor = int(rng.integers(geo.n_decor_min, geo.n_decor_max + 1))
        self.decor = []
        for _ in range(n_decor):
            color = str(rng.choice(app.decor_colors))
            self.decor.append(self.place_entity(Ball(color=color)))

        self.place_agent()

        self.last_layout = {"size": round(size, 3), "n_decor": n_decor}
        self.last_appearance = {
            "wall_tex": wall_tex,
            "floor_tex": floor_tex,
            "ceil_tex": ceil_tex,
        }

    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)
        if self.near(self.box):
            reward += self._reward()
            termination = True
        return obs, reward, termination, truncation, info

    def state_metadata(self) -> dict:
        """Everything the brief wants recorded as metadata (section 9),
        kept out of the observation actually handed to the model."""
        return {
            "agent_position": [float(v) for v in self.agent.pos],
            "agent_orientation": float(self.agent.dir),
            "goal_position": [float(v) for v in self.box.pos],
        }
