# Phase 2: plain V-JEPA-style pretraining on the MiniGrid pilot corpus

## Context

Phase 1 shipped the environment and a verified 500-episode pilot dataset.
The model itself was always deferred to a later phase -- this is that
phase, run on a GPU box (Tesla T4) rather than the CPU box the environment
was built on. The pilot dataset was regenerated here first (same generator,
same stats as the original run: 100% obstacle motion, 97% key pickup, 84%
door-open, 93% blocked-move, 54% goal-reached -- see README's "Pilot
results").

Per explicit direction: this first model is **plain V-JEPA-style
self-supervised pretraining on frames only** -- masked spatiotemporal
prediction, no action input to the predictor. Action-conditioning
(V-JEPA-2-AC style) is a later phase, not this one. The goal here is a
**small sanity run**: enough signal (loss curve, a collapse diagnostic, and
a linear probe against ground-truth agent position) to decide whether
scaling up data generation is worth it -- not a serious first pretraining
attempt.

## Design

- **Data**: `model/dataset.py`'s `ClipDataset` decodes every episode's
  `frames.mp4` once into RAM (~1.9GB for the full pilot -- small enough to
  skip a streaming loader) and slices overlapping clips via the
  already-written but previously-unused `data_gen/clip_extractor.extract_clips`
  (`clip_len=16, stride=8`). Train/val split is **by episode** (every 10th
  sorted `episode_id` -> val), so overlapping clips from one episode never
  straddle the split.
- **Tokenization**: `PatchEmbed3D` (`model/jepa.py`) -- a `Conv3d` tubelet
  embed, kernel/stride `(2, 16, 16)` on 128x128 frames. A 16-frame clip
  becomes 8 temporal x 8x8 spatial = 512 tokens. Fixed 3D sin-cos positional
  embedding (temporal/H/W each get a third of the embedding dim).
- **Encoder**: small ViT, depth=6, dim=192, heads=6 (~2.7M params) --
  standard pre-LN MHSA+MLP blocks.
- **Masking**: `MultiBlockMask` samples ~3-4 random rectangular blocks in
  the 8x8 spatial grid (same blocks at every temporal tubelet) until ~60%
  of tokens are masked, then corrects the count to exactly the target so
  every batch element has the same number of masked/visible tokens (needed
  to batch the context encoder's variable-length visible-token sequences).
- **Target/predictor**: `target_encoder` is an EMA copy of the context
  encoder (fixed momentum 0.998, stop-gradient), run on the *full* unmasked
  clip. `Predictor` (depth=4, dim=128) takes context-encoder output at
  visible positions + mask tokens with positional embeddings at masked
  positions, self-attends, predicts the target encoder's representation at
  masked positions. Loss: smooth-L1 between prediction and
  target (`.detach()`), masked positions only.
- **Collapse diagnostic**: per-epoch, the std of target-encoder embeddings
  across the batch is logged (`target_std` in `runs/<id>/log.jsonl`) -- the
  known JEPA failure mode is representations collapsing to a near-constant
  vector, which a decreasing loss alone wouldn't reveal.
- **Probe**: `model/probe.py` freezes the trained context encoder, mean-pools
  its tokens per clip, and fits a closed-form linear regressor from that
  embedding to the agent's grid position (`meta.json`) at the clip's last
  frame. Compared against an identically-shaped **random-init** encoder as
  baseline -- a meaningfully lower held-out MSE for the trained encoder is
  the actual justification for scaling up data generation, not the loss
  curve by itself.

## Environment note (GPU box specific)

This box has a system-wide CUDA 13.2 install under `/usr/local/cuda/lib`,
present in the global `LD_LIBRARY_PATH` for every shell. That system
install's `libcudnn_engines_precompiled` plugin requires `libcublasLt.so.12`,
which doesn't exist anywhere on the box (only `.so.13` does, both system and
pip-bundled) -- and it gets picked up by torch's cudnn loader *ahead of* the
correctly-paired pip-bundled cudnn/cublas libraries (torch's cudnn frontend
dlopens engine plugins by their literal fully-versioned filename, which the
pip wheel doesn't ship, so the pip-only scenario would just skip that
optional engine -- but the system path happens to have a same-numbered file
that isn't actually compatible, so it's found and crashes instead of being
skipped). Symptom: any conv-family CUDA op (`nn.Conv2d`/`Conv3d`, so any ViT
patch-embed) crashes with `Invalid handle. Cannot load symbol
cublasLtGetVersion` -- plain matmul/Linear/attention ops are unaffected.
Fix: unset `LD_LIBRARY_PATH` for the training process (`env -u
LD_LIBRARY_PATH python model/...`) so the pip-bundled libraries are used
throughout. This affects any torch conv op on this box in any conda env,
not just this project's `jepa` env.

## What's reused unchanged

`data_gen/clip_extractor.py` (previously written, unused until now),
`dataset/pilot/manifest.jsonl` and each episode's `frames.mp4`/`meta.json`.
No changes to `data_gen/` or `env_wrapper/`.

## Verification

1. `model/smoke_test.py`: random-batch forward/backward, finite loss,
   nonzero context-encoder gradients, EMA step moves target-encoder params.
2. 2-epoch dry run on the real pilot data to catch dataset/shape bugs before
   spending a full run's wall-clock.
3. Full sanity run (30 epochs); train/val loss should trend down and
   `target_std` should stay well above a collapsed near-zero value.
4. `model/probe.py`: trained-encoder linear-probe MSE on held-out agent
   position vs. a random-init-encoder baseline.
5. Results recorded below and in `README.md`.

## Results (2026-09-16)

Dry run (2 epochs, batch 16): train_loss 0.112 -> 0.037, val_loss 0.042 ->
0.034, `target_std` stayed in a healthy 0.41-0.50 range (no collapse) --
confirmed the pipeline end-to-end before committing GPU time to the full
run.

**Run 1** (30 epochs, batch 32, lr=3e-4 peak, 3570 train / 419 val clips,
2.67M-param encoder): loss dropped cleanly for the first ~6 epochs (val
0.19 -> 0.033) then drifted upward for the rest of training even as the
cosine schedule decayed lr toward 0, ending at val_loss 0.066 -- worse than
epoch 6. `target_std` grew monotonically the whole run (0.34 -> 0.87), i.e.
no collapse, but no stabilization either. Diagnosis: peak lr (3e-4) too
aggressive for this model/data size. Linear probe on the final checkpoint:
**trained encoder MSE 26.66 vs. random-init baseline 1.33 -- trained was
~20x *worse*.**

**Run 2** (same setup, lr=1e-4 peak, plus `train.py` now tracks and saves
the best-val checkpoint separately from the final one): visibly more
stable -- loss decreased smoothly to a true minimum at epoch 8 (val_loss
0.0291), then oscillated in the 0.03-0.05 band for the remaining 21 epochs
without a clear further improving trend. `target_std` still grew
monotonically throughout (0.34 -> 0.83) -- same pattern, gentler. Linear
probe on `checkpoint_best.pt` (epoch 8): **trained encoder MSE 2.30 vs.
random-init 0.65 -- trained was still ~3.5x worse**, a much smaller gap
than run 1 but the same direction.

**Conclusion: the probe criterion this plan set (trained encoder should
beat random-init on the position probe) was not met in either run.** The
masked-prediction loss itself is clearly learning something (it drops far
below what a random/untrained target would give, and doesn't collapse),
but that something isn't linearly decodable agent position from a
mean-pooled clip embedding, at least not yet, at this scale. Plausible
reasons, undetermined which dominates: (a) 2.7M params / ~3.6k clips / 30
epochs may simply be below the scale where JEPA-style pretraining produces
linearly-probable position information -- the original V-JEPA work probes
much larger pretrained models; (b) mean-pooling across all 512 tokens
(including ones far from the agent, and across all timesteps) may dilute
whatever position-relevant signal exists in a token local to the agent;
(c) masked *frame* prediction may preferentially capture appearance/layout
regularities (obstacle colors, wall/door positions) rather than the
agent's own state, since the agent is a small part of the scene. This is a
genuine, not-yet-resolved open question -- not a pipeline bug (the
pipeline itself -- data loading, masking, EMA, no-collapse -- is verified
working). Recorded honestly rather than as a pass; see README's "Phase 2
pilot-scale run" section for the same numbers.

## Scale-up to 5,000 episodes (2026-09-16)

Regenerated a 10x larger corpus (`dataset/scale5k/`, 5000 episodes,
199.9MB, seeds 1000-5999 so no overlap with the 500-episode pilot's seeds
0-499) with `scripts/generate_pilot.py` -- dynamics stats matched the
pilot closely (83% door-open, 97% key pickup, 100% moving obstacles, 52%
goal-reached, 93% blocked-move), confirming quality held at 10x scale.

**Hit a real memory ceiling first.** `model/dataset.py`'s original
`ClipDataset` decoded the entire corpus into RAM at construction (fine at
pilot scale, ~1.9GB) -- at 5000 episodes that's ~19GB decoded, and this box
only has 15GB total RAM. Caught it mid-run (RSS climbing past 8GB and
still growing, no swap configured) and killed the job before it got
OOM-killed uncontrolled. Rewrote `ClipDataset` as an `IterableDataset` that
decodes in shuffled chunks of `chunk_size` episodes (block shuffle, not a
global shuffle) and drops each chunk's frames before moving to the next --
peak memory now stays bounded regardless of corpus size, verified via
`free -h` during a run (stayed ~5-6GB used throughout, vs. climbing
unbounded before). This also made per-epoch mp4 decoding a recurring cost
instead of a one-time cost, so added `num_workers=3` to the `DataLoader`
calls in `train.py`/`probe.py` to parallelize decode across this box's 4
CPU cores -- confirmed via a pilot-scale dry run that this brought epoch
time back down to the original in-RAM speed (49.5s), i.e. decode is fully
hidden behind GPU compute at this batch size.

**Training** (15 epochs, batch 32, lr=1e-4 peak, 36533 train / 4137 val
clips, same 2.67M-param encoder, ~550s/epoch -> ~2.3hr total): the same
drift pattern as the pilot runs, now milder and it recovers by the end --
val_loss dropped to 0.0421 by epoch 1, drifted up to 0.0618 by epoch 3,
then declined smoothly as the cosine schedule decayed lr, finishing at
0.0412 (epoch 14, the best of the run). `target_std` again only grew
(0.41 -> 0.85, plateauing in the second half) -- no collapse, and this
growth pattern now looks like a structural property of this EMA/smooth-L1
setup rather than an artifact of one bad lr choice, since it showed up
again even at the lower lr and 10x the data. Not investigated further this
round -- flagged as a real open item if scaling further.

Linear probe on `checkpoint_best.pt` (epoch 14): **trained encoder MSE
1.08 vs. random-init 0.55 -- trained still worse, but the gap closed
substantially**: 20x worse at 500 episodes (run 1, before the lr fix),
3.5x worse at 500 episodes (run 2, after the lr fix), **2x worse at 5,000
episodes**. The gap shrinking monotonically as data scales up is itself
the signal worth acting on -- it suggests data volume is a real factor,
not just a red herring, even though this run alone still doesn't clear the
bar this plan set. Doesn't yet distinguish between "needs more data" and
"needs more training/bigger model at this data size" as the dominant
lever -- both remain plausible.

## Root-caused and fixed the training-drift pattern (2026-09-16)

User pushed back on the recurring "drops, then drifts worse, then
partially recovers as lr decays" loss pattern -- correctly: that's not
what healthy training looks like, and it showed up identically at 500,
5,000, and 10,000 episodes and at two different peak lrs, so it was never
going to be fixed by more data or more lr tuning. Root-caused it properly
instead of continuing to scale blindly.

**Comparison against the actual V-JEPA2 recipe** surfaced two real
simplifications in this implementation: mask ratio was 60% (V-JEPA masks
~90%, leaving very little visible context -- an easier, less informative
task at 60%) and the EMA target used a **fixed** momentum (0.998) instead
of V-JEPA's momentum **schedule** (ramping e.g. 0.996 -> 1.0 over
training). The fixed-momentum gap was the leading suspect for the drift
itself, since target_std grew monotonically in every run regardless of
scale or lr.

**That hypothesis was wrong**, and the ablation that disproved it is worth
recording because chasing it further would have wasted real time:
1. Set `--ema-momentum-start 1.0 --ema-momentum-end 1.0` (target encoder
   provably frozen -- verified per-step, bit-for-bit, that
   `target_encoder`'s parameters never move under this setting).
2. `target_std` **still grew** (0.34 -> ~0.50 over a few epochs) even
   though the target encoder's own weights were frozen the entire time.
3. Traced it to `patch_embed` (the trainable Conv3d tubelet projection,
   shared by both the context and target paths): its raw output std grew
   from 0.27 -> 1.10 over the first ~40 steps and plateaued. Nothing
   constrained its output scale, so the *input* to the frozen target
   encoder was non-stationary -- that alone fully explains a frozen
   encoder's output std growing, no EMA instability required.
4. Fix: added `self.patch_norm = nn.LayerNorm(cfg.encoder_dim)` applied to
   `patch_embed`'s output before adding the positional embedding
   (`model/jepa.py`, `JEPA.__init__`/`forward`). Re-ran the same frozen-
   target ablation: `raw_tokens.std()` now stays pinned at ~1.0 instead of
   climbing to 1.1, and `target_std` settles into a bounded 0.27-0.36
   range instead of climbing unboundedly.
5. Re-ran real (non-frozen) training with the fix + the mask-ratio-0.9 and
   momentum-schedule changes together (10 epochs, pilot data, lr=1e-4):
   val_loss now decreases **every single epoch** (0.309 -> 0.162 -> 0.110
   -> 0.096 -> 0.087 -> 0.082 -> 0.079 -> 0.078 -> 0.077 -> 0.077) --
   exactly the "several drops before flattening" shape expected of healthy
   training, and the complete opposite of every prior run's drift pattern.

**A process note, since it nearly produced a wrong conclusion**: the
*first* attempt at this frozen-target ablation was run while a previous
`scale10k` training job was still finishing in the background on the same
box. That job crashed with `DataLoader worker...killed by signal: Killed`
(this box has only 15GB RAM, no swap -- two concurrent training jobs' worth
of persistent DataLoader workers exceeded it), and the concurrent ablation
run showed target-encoder norm-weight values jumping around even though a
truly frozen target should be bit-identical across every step. Re-running
the exact same ablation alone (no concurrent job) still showed the jump,
which briefly looked like a real non-determinism bug -- until per-step
instrumentation (`.clone()` + explicit sync each step) made it disappear
entirely across multiple clean reruns. That inconsistency was itself the
clue: it wasn't a per-step accounting bug, it was real, gradual growth in
`patch_embed`'s output landing on the actual mechanism once measured
directly (see point 3 above) rather than inferred from noisy end-of-epoch
parameter snapshots. Lesson: don't run two GPU/RAM-heavy jobs on this box
at once, and when a diagnostic reading looks paradoxical, measure the
actual quantity in question directly rather than trusting a proxy.

Not yet done: rerunning the probe (agent-position linear regression) with
this fixed model at the 5,000/10,000-episode scale already generated, to
see whether the healthier training dynamics also finally clear the
probe-vs-random-init bar that no run had cleared before this fix.
