"""Custom MiniGrid environment purpose-built for dynamics-rich JEPA
pretraining video: a room split by a wall with a locked door + matching
key, a goal on the far side, and a handful of independently-moving
obstacles. See docs/phase1-plan.md for why this replaced the MiniWorld
OneRoom wrapper (empty room, camera-motion-only, not a meaningful test bed
for action-conditioned post-training).

Modeled directly on two of MiniGrid's own reference envs rather than
invented from scratch:
- the split-room/door/key/goal layout mirrors `minigrid.envs.doorkey.DoorKeyEnv`
- the independent obstacle motion is copied from
  `minigrid.envs.dynamicobstacles.DynamicObstaclesEnv.step` (each obstacle
  attempts a random reposition in its own 3x3 neighborhood every step, via
  the same rejection-sampling `place_obj` the base class already uses for
  initial placement)

Deliberate difference from `DynamicObstaclesEnv`: we do *not* copy its
obstacle-collision termination. An obstacle blocks the agent's forward
move exactly like a wall (free, from `WorldObj.can_overlap()` defaulting
to False) -- a frequent, benign "blocked" transition rather than a rare
terminal failure, which is the point: MiniGrid's own base `step()` (not
overridden by us) already makes a blocked-vs-successful move visibly
distinguishable, we just don't want it to end the episode.
"""
import os
from operator import add

# must be set before minigrid/pygame is imported -- lets rendering work with
# no real display at all (verified on this box), unlike MiniWorld's pyglet
# which needed a real GLX connection (Xvfb) even for rgb_array rendering.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball, Door, Goal, Key, Wall
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper

from env_wrapper.config import WrapperConfig


def _gen_mission() -> str:
    return "pick up the key, open the door, get to the goal"


class MiniGridJepaEnv(MiniGridEnv):
    def __init__(self, config: WrapperConfig = None, max_steps: int = None, **kwargs):
        self.jepa_config = config or WrapperConfig()
        size = self.jepa_config.geometry.grid_size
        max_steps = max_steps or self.jepa_config.episode.max_steps

        # populated by _gen_grid, read back by state_metadata()/generators
        self.last_layout: dict = {}
        self.last_appearance: dict = {}
        self.obstacles = []

        mission_space = MissionSpace(mission_func=_gen_mission)
        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=max_steps,
            see_through_walls=True,
            highlight=False,  # full, evenly-lit frame -- no partial-observability tint
            tile_size=self.jepa_config.episode.tile_size,
            render_mode="rgb_array",
            **kwargs,
        )

        # left, right, forward, pickup, drop, toggle -- excludes the unused
        # "done" action (Actions enum order, core/actions.py)
        self.action_space = type(self.action_space)(self.actions.toggle + 1)

    def _gen_grid(self, width, height):
        geo = self.jepa_config.geometry
        colors = self.jepa_config.appearance.colors
        rng = self.np_random

        wall_color = str(rng.choice(colors))
        split_color = str(rng.choice(colors))
        door_color = str(rng.choice(colors))

        self.grid = Grid(width, height)
        self.grid.horz_wall(0, 0, width, obj_type=lambda: Wall(color=wall_color))
        self.grid.horz_wall(0, height - 1, width, obj_type=lambda: Wall(color=wall_color))
        self.grid.vert_wall(0, 0, height, obj_type=lambda: Wall(color=wall_color))
        self.grid.vert_wall(width - 1, 0, height, obj_type=lambda: Wall(color=wall_color))

        split_idx = int(rng.integers(2, width - 2))
        self.grid.vert_wall(split_idx, 0, obj_type=lambda: Wall(color=split_color))

        self.goal_pos = (width - 2, height - 2)
        self.put_obj(Goal(), *self.goal_pos)

        self.place_agent(size=(split_idx, height))

        door_idx = int(rng.integers(1, height - 1))
        self.door = Door(door_color, is_locked=True)
        self.put_obj(self.door, split_idx, door_idx)

        self.key = Key(door_color)
        self.place_obj(self.key, top=(0, 0), size=(split_idx, height))

        n_obstacles = int(rng.integers(geo.n_obstacles_min, geo.n_obstacles_max + 1))
        self.obstacles = []
        obstacle_colors = []
        for _ in range(n_obstacles):
            color = str(rng.choice(colors))
            obstacle_colors.append(color)
            obstacle = Ball(color=color)
            self.obstacles.append(obstacle)
            self.place_obj(obstacle, max_tries=100)

        self.mission = _gen_mission()

        self.last_layout = {
            "grid_size": width,
            "split_idx": split_idx,
            "door_idx": door_idx,
            "n_obstacles": n_obstacles,
        }
        self.last_appearance = {
            "wall_color": wall_color,
            "split_color": split_color,
            "door_color": door_color,
            "obstacle_colors": obstacle_colors,
        }

    def step(self, action):
        # Move obstacles first, independent of the agent's action -- see
        # module docstring. Copied from DynamicObstaclesEnv.step (without
        # its collision-termination block).
        for obstacle in self.obstacles:
            old_pos = obstacle.cur_pos
            top = tuple(map(add, old_pos, (-1, -1)))
            try:
                self.place_obj(obstacle, top=top, size=(3, 3), max_tries=100)
                self.grid.set(old_pos[0], old_pos[1], None)
            except Exception:
                pass

        return super().step(action)

    def state_metadata(self) -> dict:
        """Everything the brief wants recorded as metadata, kept out of the
        observation actually handed to the model. Includes door/obstacle
        state specifically so the verification step (docs/phase1-plan.md)
        can check that door-open and obstacle-move events actually land in
        the recorded data, not just in theory."""
        carrying = None
        if self.carrying is not None:
            carrying = {"type": self.carrying.type, "color": self.carrying.color}
        return {
            "agent_position": [int(v) for v in self.agent_pos],
            "agent_orientation": int(self.agent_dir),
            "carrying": carrying,
            "door_is_open": bool(self.door.is_open),
            "door_is_locked": bool(self.door.is_locked),
            "goal_position": [int(v) for v in self.goal_pos],
            "obstacle_positions": [
                [int(v) for v in ob.cur_pos] for ob in self.obstacles
            ],
        }


def make_env(config: WrapperConfig = None):
    """`MiniGridJepaEnv` wrapped so `reset()`/`step()` return a bare full
    top-down RGB frame as `obs` (not MiniGrid's default symbolic partial
    Dict observation) -- matches the contract data_gen/episode_generator.py
    expects (mirrors how MiniWorldEnv returned the render directly).
    `last_layout`/`last_appearance`/`state_metadata()`/`jepa_config` all
    remain reachable on the wrapped object via gymnasium's attribute
    delegation to the inner env.
    """
    config = config or WrapperConfig()
    env = MiniGridJepaEnv(config=config)
    env = RGBImgObsWrapper(env, tile_size=config.episode.tile_size)
    env = ImgObsWrapper(env)
    return env
