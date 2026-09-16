"""Config knobs for the MiniWorld JEPA wrapper.

Kept as plain dataclasses (not argparse/hydra) since the whole surface area
is small enough that a config file would be more machinery than the config.
"""
from dataclasses import dataclass, field


@dataclass
class GeometryConfig:
    """Room size range. OneRoom's own defaults are size=10 (OneRoomS6 uses 6)."""

    size_min: float = 6.0
    size_max: float = 14.0
    # number of decorative (irrelevant) objects scattered in the room
    n_decor_min: int = 0
    n_decor_max: int = 3


@dataclass
class AppearanceConfig:
    """Texture *families* to sample from per episode.

    Each name must be a valid prefix under miniworld/textures/ (miniworld
    itself then randomizes which numbered variant, e.g. wood_1 vs wood_2,
    within the family we picked -- see env_wrapper/wrapper.py).
    """

    wall_textures: tuple = (
        "wood",
        "brick_wall",
        "concrete",
        "drywall",
        "stucco",
        "marble",
        "cinder_blocks",
    )
    floor_textures: tuple = (
        "floor_tiles_bw",
        "wood_planks",
        "concrete",
        "marble",
    )
    ceiling_textures: tuple = (
        "ceiling_tiles",
        "ceiling_tile_noborder",
        "drywall",
        "concrete_tiles",
    )
    # colors for decorative clutter objects; deliberately excludes red so
    # clutter is never confusable with the red goal box
    decor_colors: tuple = ("green", "blue", "purple", "yellow", "grey")


@dataclass
class EpisodeConfig:
    obs_size: int = 128
    max_steps: int = 100
    # probability of repeating the previous action; pure-uniform-random
    # turn/forward looks jittery, this biases the walk to look purposeful
    momentum: float = 0.6


@dataclass
class WrapperConfig:
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    appearance: AppearanceConfig = field(default_factory=AppearanceConfig)
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
