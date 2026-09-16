# MiniWorld → JEPA Phase 1: environment wrapper + pilot video corpus

## Context

You want to deliberately split this into two phases and commit only to Phase 1 right now: build a MiniWorld-based environment wrapper, generate a **pilot** corpus of short unlabeled videos, and verify the generation pipeline (diversity, storage, timing) *before* deciding whether to scale to 10k+ clips or start writing the JEPA model itself. That staging (pilot → inspect → scale only if the pipeline earns it) is explicit in your own brief, so this plan stops at "pilot generated and inspected," not at "JEPA model trained."

This is a new, from-scratch project — nothing MiniWorld/JEPA-related exists on disk yet. It's not part of the `ai-build-and-learn` public stream repo (that repo's `topics/v-jepa` is a *different*, already-shipped project: probing a **pretrained** V-JEPA 2 checkpoint on Kinetics clips via Flyte/DGX-Spark. This project trains a **small JEPA-style model from scratch** on **procedurally generated** MiniWorld video — different goal, different infra). It fits the pattern of `car_dreamer/dreamer-car-nav`: a personal project living as a sibling directory under `/home/ubuntu/world_models/`, own conda env, own `notes/` journal.

Two environment facts constrain the design:
- **No GPU is currently attached to this box** (`nvidia-smi` fails; only Amazon's virtual VGA device shows in `lspci`). Irrelevant for Phase 1 (MiniWorld rendering is cheap, CPU-only is fine) but means model training later will need either a GPU to show up or a CPU-friendly toy model.
- **Disk is at 92% (11 GB free)**. The pilot budget and storage format need to respect that headroom explicitly, not just "add videos and see."

## Where it lives

New sibling directory: `/home/ubuntu/world_models/miniworld-jepa/`
New dedicated conda env: `miniworld` (python 3.11, matching the "one env per project" house convention — `t3d` and `dreamer` stay untouched).

```
miniworld-jepa/
  env_wrapper/
    __init__.py
    config.py         # dataclasses: GeometryConfig, AppearanceConfig, EpisodeConfig
    wrapper.py         # MiniWorldJepaEnv: subclasses/wraps miniworld.envs.OneRoom
    appearance.py       # texture/color/lighting randomization presets
    policy.py           # random-walk-with-momentum action policy (pure uniform-random
                         # turn/forward looks too jittery for "natural" trajectories)
  data_gen/
    episode_generator.py  # one episode = randomize → reset → roll out policy → record
    clip_extractor.py      # overlapping fixed-length clips from one long episode
    storage.py             # write frames (mp4 via imageio-ffmpeg) + metadata (jsonl)
  scripts/
    smoke_test.py          # launch OneRoom headless, step, save one PNG — proves
                            # Xvfb + pyglet + mesa softpipe actually renders on this box
    generate_pilot.py       # CLI: N episodes, writes dataset/pilot/
    inspect_dataset.py      # sample grid across appearance/layout combos, print
                            # disk usage + timing, basic diversity sanity checks
  dataset/                 # gitignored — pilot/ output lives here
  notes/                    # gitignored: journal.md + concepts/*.md, same convention
                            # as dreamer-car-nav
  docs/
    phase1-plan.md          # versioned copy of the agreed spec
  requirements.txt
  README.md
  .gitignore
```

No Flyte. The v-jepa topic's Flyte pipeline is real infra (custom container images, a local registry, an arm64/DGX-Spark-pinned setup) that doesn't exist on this box and would be pure overhead for a local data-gen script — plain Python matches what `dreamer-car-nav` already does successfully.

## Headless rendering

MiniWorld renders through Pyglet/OpenGL (unlike `dreamer-car-nav`'s pure-Pillow rendering, which needs no display at all) — it needs a real GL context even for `render_mode="rgb_array"`. Confirmed on this box: `libGLX_mesa`/`libGL.so.1` are present, and `Xvfb`/`xvfb-run` are installed, so Mesa's software rasterizer (llvmpipe) can back a virtual framebuffer with no GPU. The existing `T3D-car-navigation/start_display.sh` Xvfb+fluxbox+VNC stack is for **live GUI viewing** and is more than this needs. For batch corpus generation, just wrap each script:

```bash
xvfb-run -a .../envs/miniworld/bin/python scripts/generate_pilot.py ...
```

`smoke_test.py` is step one specifically to prove this combination actually renders a frame on this box before anything else is built on top of it — if it doesn't, everything downstream stalls, so it's worth confirming first rather than assuming.

## Environment choice and wrapper design

Start from MiniWorld's built-in `OneRoom` (`miniworld.envs.oneroom.OneRoom`), not a from-scratch env — it already has a rectangular room, a configurable `size`, and a red-box goal object baked into the default task, which covers most of section 3's "Geometry" and "Goal" requirements for free. The wrapper's actual new work is:

- **Appearance randomization** — MiniWorld rooms take per-surface textures (`wall_tex`, `floor_tex`, `ceil_tex`) and its bundled texture set (`miniworld/textures/`) has enough materials (wood, brick, concrete, carpet, grass, etc.) for real "same room, different look" pairs; colors/lighting come from `Room`/`MiniWorldEnv` params. Exact override points (`_gen_world`, `Room` constructor kwargs) get confirmed by reading the installed package source in step 1 of implementation, not guessed now.
- **Agent start pose + goal placement randomization** — expose explicit control over `place_agent`/`place_entity` calls with a seeded RNG so we can log the exact configuration used (needed for the metadata in section 9).
- **Action policy** — pure-uniform-random `{turn_left, turn_right, move_forward}` produces visually jittery, non-purposeful trajectories. Use a simple momentum-biased random walk (weight toward repeating the previous action) so trajectories look like plausible short walks rather than noise — still fully unsupervised/no RL involved.
- **Metadata capture** — `env.agent.pos`, `env.agent.dir`, and the goal entity's position are read directly off the MiniWorld env each step; no simulator modification needed, matching the "wrapper, don't fork" instruction.

## Data generation pipeline

- One **episode** = randomize geometry config + appearance config + start/goal → roll out the momentum-biased policy for ~100+ steps → record every frame (128×128 RGB, fixed camera, per section 5) + per-step metadata (`episode_id, environment_id, layout_id, appearance_id, timestep, agent_position, agent_orientation, goal_position, action`).
- **Clips** are extracted from each episode as overlapping fixed-length windows (default 32 frames, configurable, per section 6) — this is a pure array-slicing step in `clip_extractor.py`, not a second simulator pass.
- **Storage**: each episode's frames as one small mp4 (via `imageio`/`imageio-ffmpeg`, already need pyav-free ffmpeg on this box or fall back to imageio's ffmpeg plugin) plus one `.jsonl` metadata sidecar. mp4 keeps the pilot small under the 11 GB constraint versus raw PNG-per-frame dumps.
- **Splits** are by `layout_id`/`appearance_id`, not by frame or clip (section 11) — implemented as a manifest step (a CSV/JSON index of which episode belongs to train/val/test-structural/test-appearance), never by physically separating files.

## Pilot run + inspection (the actual deliverable of this plan)

- `generate_pilot.py --n-episodes 500 --episode-len 100` → writes `dataset/pilot/`, printing running disk usage so we stop early if projections blow past the 11 GB headroom.
- `inspect_dataset.py` → renders a contact-sheet grid sampling across different appearance/layout combinations (the "same world, different appearance" check from section 8), reports total disk size, total generation wall-time, frame count, and a couple of trivial sanity numbers (e.g. distribution of episode lengths, action histogram) — enough to make the "verify generation / inspect diversity / estimate storage & time" pilot checklist from your brief concrete and checkable, without building any probes or model code yet.

Explicitly out of scope for this plan (per your own staging — "scale when the experiment tells us to," and Phase 1's model work is a separate step): the 10k–20k clip corpus, dataset-split-family holdouts for appearance generalization, the JEPA model itself, and any probes/evaluation. Those are the next increment, after this pilot is generated and actually looked at.

## Verification

1. `xvfb-run -a .../envs/miniworld/bin/python scripts/smoke_test.py` renders and saves one non-blank PNG from `OneRoom` — proves the headless GL path works on this box.
2. `generate_pilot.py --n-episodes 20` (small smoke run) completes, produces the expected file layout, and metadata round-trips (positions/orientations change sensibly frame to frame, actions match the policy's action space).
3. Full 500-episode pilot run completes within a sane wall-clock time and disk budget (both printed by the script).
4. `inspect_dataset.py`'s contact sheet visibly shows appearance variety (different textures/colors) across otherwise-similar layouts, confirming the "same state, different pixels" property the brief cares about — this is a manual look at the output, not an automated assertion.
