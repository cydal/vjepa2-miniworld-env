"""Config knobs for the MiniGrid JEPA wrapper.

Kept as plain dataclasses (not argparse/hydra) since the whole surface area
is small enough that a config file would be more machinery than the config.
"""
from dataclasses import dataclass, field


@dataclass
class GeometryConfig:
    """Grid is a fixed 8x8 for this iteration (keeps the frame a fixed
    128x128 with no camera randomization, per the brief). Layout diversity
    instead comes from where the splitting wall/door/key/goal/obstacles
    land, which varies every episode. Grid-size variation is a deliberate
    fast-follow, not in this iteration -- see docs/phase1-plan.md.
    """

    grid_size: int = 8
    n_obstacles_min: int = 2
    n_obstacles_max: int = 4


@dataclass
class AppearanceConfig:
    """MiniGrid's palette is a 6-way categorical, not continuous textures --
    see core/constants.py COLOR_NAMES. Door/key color must match for the
    episode to be solvable, so color is functionally load-bearing here, not
    just decorative.
    """

    colors: tuple = ("red", "green", "blue", "purple", "yellow", "grey")


@dataclass
class EpisodeConfig:
    tile_size: int = 16  # grid_size(8) * tile_size(16) = 128x128 frames
    max_steps: int = 100
    # probability of repeating the previous movement action; pure-uniform
    # -random turn/forward looks jittery, this biases the walk to look
    # purposeful. Pickup/toggle are never subject to momentum -- they only
    # fire reactively (see env_wrapper/policy.py).
    momentum: float = 0.6


@dataclass
class WrapperConfig:
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    appearance: AppearanceConfig = field(default_factory=AppearanceConfig)
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
