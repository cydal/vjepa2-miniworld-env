# MiniWorld → MiniGrid pivot: a dynamics-rich custom environment (Phase 1, take 2)

## Context

The MiniWorld/OneRoom pilot shipped previously (500 episodes, verified, committed) turned out to be the wrong substrate: it's an empty room where the only thing that ever changes is the camera's own pose. That's not a meaningful test bed for JEPA-style pretraining, and it's specifically weak for what this is actually building toward: action-conditioned post-training (V-JEPA-2-AC style), where the whole point is that identical visual context plus a different action should produce a different, learnable future. Camera-pan optic flow doesn't exercise that; an agent's action needs to visibly change the *state of the world*, not just its own viewpoint.

Clarifying-question answers that shape this plan:
- Dynamics that matter: **independent world dynamics** (things change without the agent acting) and **collision/blocked motion** (an action's effect is state-dependent — sometimes it works, sometimes it's blocked).
- Engine: open to swapping away from MiniWorld if it serves the dynamics better.
- This is explicitly feeding **action-conditioned post-training** — "sensible transitions" means actions with distinguishable, non-trivial consequences.
- Scope: build one concrete next iteration now, not a broader survey.

Investigated MiniWorld's actual engine capabilities first (not guessed): confirmed it never calls a per-step update on non-agent entities at all (`Entity.step()` is defined but nothing invokes it), and its `toggle` action is declared in the `Actions` enum but has no handler anywhere in `MiniWorldEnv.step()` — dead code. Getting independent dynamics or working doors out of MiniWorld means writing new engine-level simulation code, not configuring existing code.

Verified an alternative, **MiniGrid** (Farama Foundation's sibling project, 2D top-down grid world), directly on this box:
- Installs cleanly, renders headless via `SDL_VIDEODRIVER=dummy` — no Xvfb, no `libglu1-mesa`, none of the GL pain MiniWorld needed.
- `MiniGrid-Dynamic-Obstacles-8x8-v0`: obstacles visibly relocate every step even when the agent's action never moves it (confirmed by rendering 7 frames with the agent only turning — obstacles moved, agent didn't).
- `MiniGrid-DoorKey-8x8-v0`: a **real** functioning door/key mechanic — `Door.toggle()` actually checks `env.carrying` against the door's color and only unlocks on a match; MiniWorld's equivalent action is a no-op.

Decision: replace the MiniWorld-based environment with a custom MiniGrid environment purpose-built to contain both requested dynamics classes at once.

## Design: `MiniGridJepaEnv`

One room split by an internal wall with a locked door (random color) + matching key (same color) placed somewhere in the near side, a goal tile in the far side, and a handful of dynamic obstacles (`Ball` entities) scattered through the room.

- **Actions**: `Discrete(6)` = `left, right, forward, pickup, drop, toggle` (`core/actions.py` enum; excludes the unused `done` action). Needing `pickup`/`toggle` for the door/key mechanic is *why* the action set is richer than MiniWorld's movement-only 3.
- **Per-episode randomization**: happens inside `_gen_grid(width, height)`, which `reset()` calls *after* seeding `self.np_random` (`minigrid_env.py:119-157`), so the existing `self._rand_color()` / `self._rand_int()` / `place_obj()` rejection-sampling helpers give us per-episode-seeded placement, same pattern as MiniWorld's `_gen_world`. Randomize: wall-split position, door/key color (matched), goal position, obstacle count and starting positions.
- **Independent dynamics**: copy the obstacle-relocation loop from `DynamicObstaclesEnv.step()` (`dynamicobstacles.py:136-156`, each obstacle attempts a random reposition in its 3x3 neighborhood via `place_obj`) into our own `step()` override, run *before* calling `super().step(action)`. This is the exact mechanism already proven to work — obstacles move regardless of what the agent does.
- **Collision semantics — deliberately different from stock `DynamicObstaclesEnv`**: do **not** copy its obstacle-collision termination (`dynamicobstacles.py:161-165`, which ends the episode on agent/obstacle collision). Here, walking into an obstacle (or the locked door, or the wall) just blocks the move like any wall — a benign, frequent, clearly-labeled "blocked" transition rather than a rare failure event. This is what actually gives us the "collision & blocked motion" dynamics class as common, learnable data rather than an edge case.
- **Rendering**: full top-down frame, not the agent-egocentric partial view — `render_mode="rgb_array"` / `get_frame(agent_pov=False, tile_size=...)` (`minigrid_env.py:668-785`). Fixed grid size **8x8 with `tile_size=16` → exactly 128x128px**, no camera randomization, consistent with the original brief's "vary the world, not the camera." Grid-size variation is a plausible fast-follow, not in this iteration — layout diversity for now comes from wall-split position, door/key/goal/obstacle placement, and obstacle count, which is already substantial.
- **Appearance randomization**: MiniGrid's palette is a 6-way categorical (`red, green, blue, purple, yellow, grey`), not MiniWorld's continuous textures — coarser, but door/key color *must* match for the episode to be solvable, so color is functionally load-bearing here, not just decorative. That's arguably a better property for later representation probing than MiniWorld's purely-cosmetic texture swaps.

## Action policy: momentum-random + reactive pickup/toggle

Pure momentum-random movement (the existing `MomentumRandomPolicy`) would almost never wander onto the key at the right moment, or reach the door while carrying it — the corpus would rarely contain the door-opens transition that's the actual point of building this. Extend the policy: when the cell directly ahead is the key and the agent isn't carrying anything, force `pickup`; when the cell ahead is a locked door and the agent is carrying the matching key, force `toggle`; otherwise fall back to the existing momentum-biased random walk over `{left, right, forward}`. Still fully scripted/reactive, not RL — consistent with the original brief's "actions only generate trajectories, no learning involved."

## What's reused unchanged

`data_gen/episode_generator.py`, `data_gen/clip_extractor.py`, `data_gen/storage.py`, `scripts/generate_pilot.py`, `scripts/inspect_dataset.py` only depend on a small contract: `env.reset(seed=)`, `env.step(action)`, `env.last_layout`, `env.last_appearance`, `env.state_metadata()`, `env.jepa_config.episode.max_steps`. `MiniGridJepaEnv` keeps that exact contract, so these files need no changes. mp4 storage via `imageio` is unaffected — frames are still plain uint8 arrays, just from a different renderer.

## What changes

- `env_wrapper/config.py` — swap MiniWorld texture-family lists for MiniGrid's 6-color palette; geometry config becomes obstacle-count range (grid-size range deferred).
- `env_wrapper/wrapper.py` — new `MiniGridJepaEnv(MiniGridEnv)` replacing the MiniWorld-based class, per the design above.
- `env_wrapper/policy.py` — extend `MomentumRandomPolicy` with the reactive pickup/toggle rule; it now needs to inspect the cell ahead of the agent each `act()` call, so it takes the env (not just an RNG) as an argument.
- `requirements.txt` / `README.md` — swap `miniworld`/`pyglet` for `minigrid`; drop the Xvfb/`libglu1-mesa` setup instructions (not needed — verified headless via `SDL_VIDEODRIVER=dummy`).
- `scripts/smoke_test.py` — updated for the new env and action set.
- Pilot data: move the existing MiniWorld pilot to `dataset/_archive_miniworld_pilot_v1/` (kept for the journal's before/after comparison; still gitignored) and generate a fresh 500-episode MiniGrid pilot at `dataset/pilot/`.
- `notes/journal.md` — record why the pivot happened and what changed, per the standing convention.

## Verification

1. Updated `smoke_test.py`: reset, confirm the frame is exactly 128x128 and non-blank, run a scripted rollout, confirm the reactive policy actually issues `pickup`/`toggle` at the right moments in at least one short run, save sample frames.
2. Small dry run (`generate_pilot.py --n-episodes 20`): manually inspect a few `meta.json` step sequences to confirm obstacle positions change between consecutive steps even when the agent's own action was e.g. `left` (proves independent dynamics landed in the actual data, not just in theory), and that at least some episodes show `carrying → key` followed by the door's `is_open` flipping to `True`.
3. Full 500-episode pilot regenerated; rerun `inspect_dataset.py`. Contact sheet should now show colored doors/keys/obstacles, not empty rooms. Add two counts specific to this iteration's goal: how many episodes exercised a door-open event, and how many had at least one obstacle- or door-blocked move — a concrete diversity check tied to what we were actually trying to fix, not just texture/appearance variety.
