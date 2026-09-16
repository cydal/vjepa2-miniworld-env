"""Unattended pipeline: retrain on the already-generated 10k corpus with
the patch_norm fix, gate on the probe actually beating random-init (the
bar no run had cleared before that fix), and if it clears, scale up data
and run a longer training pass -- all with no human in the loop.

Launched via nohup + disown specifically so it keeps running on this box
regardless of the controlling shell/session/laptop connection. Every step
is logged to overnight_log.txt (timestamped) and the final outcome is
written to overnight_report.md. Decisions are made here, once, rather
than relying on an interactive agent staying alive to reason about
intermediate results -- see docs/phase2-plan.md for the reasoning behind
the specific thresholds/scale chosen.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MINIWORLD_PY = "/home/ubuntu/miniconda3/envs/miniworld/bin/python"
JEPA_PY = "/home/ubuntu/miniconda3/envs/jepa/bin/python"
LOG_PATH = REPO / "overnight_log.txt"
REPORT_PATH = REPO / "overnight_report.md"

# This box's system-wide CUDA 13.2 install conflicts with pip-installed
# torch's cudnn/cublas for any conv op -- see docs/phase2-plan.md /
# memory "gpu-box-cudnn-ld-library-path". Every jepa-env call needs this.
JEPA_ENV = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}


def log(msg: str):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def run(cmd, env=None, cwd=REPO):
    log(f"RUN: {' '.join(cmd)}")
    result = subprocess.run(
        cmd, cwd=cwd, env=env, capture_output=True, text=True
    )
    log(f"exit={result.returncode}")
    if result.stdout:
        log("stdout tail:\n" + "\n".join(result.stdout.splitlines()[-15:]))
    if result.returncode != 0:
        log("stderr tail:\n" + "\n".join(result.stderr.splitlines()[-30:]))
    return result


def parse_probe_output(stdout: str):
    """probe.py prints e.g. 'trained: linear probe MSE on agent_position = 1.08 ...'"""
    trained_mse, random_mse = None, None
    for line in stdout.splitlines():
        if line.startswith("trained:"):
            trained_mse = float(line.split("=")[1].split("(")[0].strip())
        elif line.startswith("random_init:"):
            random_mse = float(line.split("=")[1].split("(")[0].strip())
    return trained_mse, random_mse


def write_report(sections: dict):
    with open(REPORT_PATH, "w") as f:
        f.write("# Overnight run report\n\n")
        for title, body in sections.items():
            f.write(f"## {title}\n\n{body}\n\n")


def main():
    sections = {}
    log("=" * 60)
    log("overnight_run.py starting")

    # --- Step 1: retrain on scale10k with the patch_norm fix ---
    step1_dir = REPO / "runs" / "overnight_step1_scale10k"
    step1_dir.mkdir(parents=True, exist_ok=True)
    log("Step 1: retrain on dataset/scale10k with patch_norm fix (12 epochs, lr=1e-4)")
    r = run(
        [JEPA_PY, "-u", "model/train.py", "--data-dir", "dataset/scale10k",
         "--epochs", "12", "--lr", "1e-4", "--out-dir", str(step1_dir)],
        env=JEPA_ENV,
    )
    sections["Step 1: retrain on scale10k"] = (
        f"Exit code: {r.returncode}\n\n```\n{r.stdout[-3000:]}\n```"
    )
    if r.returncode != 0:
        sections["Outcome"] = "FAILED at step 1 (retrain). Stopped -- did not proceed to scale-up."
        write_report(sections)
        log("Step 1 failed, stopping.")
        return

    # --- Step 2: probe gate -- the bar no run had cleared before the fix ---
    log("Step 2: probe on step1's best checkpoint")
    r = run(
        [JEPA_PY, "-u", "model/probe.py", "--data-dir", "dataset/scale10k",
         "--checkpoint", str(step1_dir / "checkpoint_best.pt")],
        env=JEPA_ENV,
    )
    trained_mse, random_mse = parse_probe_output(r.stdout)
    sections["Step 2: probe gate"] = (
        f"trained_mse={trained_mse} random_mse={random_mse}\n\n```\n{r.stdout}\n```"
    )
    gate_passed = (
        r.returncode == 0
        and trained_mse is not None
        and random_mse is not None
        and trained_mse < random_mse
    )
    log(f"Gate: trained_mse={trained_mse} random_mse={random_mse} passed={gate_passed}")

    if not gate_passed:
        sections["Outcome"] = (
            "Probe gate NOT passed (trained encoder did not beat random-init on the "
            "held-out agent-position probe even with the patch_norm fix). Per plan, "
            "stopping here rather than spending a large data-generation + training "
            "budget on a foundation that still isn't clearing the bar -- this needs "
            "a person to look at it, not more scale."
        )
        write_report(sections)
        log("Gate not passed, stopping before scale-up.")
        return

    # --- Step 3: gate passed -- generate a much larger corpus ---
    # Decision (made now, not re-derived at 3am): 50,000 episodes -- 5x the
    # 10k corpus already generated. Disk (~40KB/episode -> ~2GB) and RAM
    # (streaming loader, bounded regardless of corpus size) both have
    # comfortable headroom; generation throughput (~7.25 ep/s) puts this at
    # ~2hr, leaving the rest of an overnight budget for the actual training.
    N_EPISODES = 50000
    SEED_START = 100000  # disjoint from pilot(0-499)/scale5k(1000-5999)/scale10k(20000-29999)
    data_dir = REPO / "dataset" / "scale50k"
    log(f"Gate passed. Step 3: generating {N_EPISODES} episodes -> {data_dir}")
    r = run(
        [MINIWORLD_PY, "scripts/generate_pilot.py",
         "--n-episodes", str(N_EPISODES), "--episode-len", "100",
         "--seed-start", str(SEED_START), "--out-dir", "dataset/scale50k"],
    )
    sections["Step 3: generate 50k-episode corpus"] = (
        f"Exit code: {r.returncode}\n\n```\n{r.stdout[-3000:]}\n```"
    )
    if r.returncode != 0:
        sections["Outcome"] = "FAILED at step 3 (data generation). Stopped."
        write_report(sections)
        log("Step 3 failed, stopping.")
        return

    log("Step 3b: inspect_dataset.py sanity check on scale50k")
    r = run([MINIWORLD_PY, "scripts/inspect_dataset.py", "--dataset-dir", "dataset/scale50k"])
    sections["Step 3b: dynamics check on scale50k"] = f"```\n{r.stdout}\n```"

    # --- Step 4: full training run on the 50k corpus ---
    # Decision: 10 epochs. Scale10k (72,753 train clips) took ~750s/epoch;
    # scale50k has ~5x the clips, so expect ~5x epoch time (~1hr/epoch) --
    # 10 epochs (~10hr) fits an overnight budget while giving ~4x the total
    # gradient steps of the scale10k run (10*5x clips vs 12*1x clips).
    step4_dir = REPO / "runs" / "overnight_step4_scale50k"
    step4_dir.mkdir(parents=True, exist_ok=True)
    log("Step 4: full training run on dataset/scale50k (10 epochs, lr=1e-4)")
    r = run(
        [JEPA_PY, "-u", "model/train.py", "--data-dir", "dataset/scale50k",
         "--epochs", "10", "--lr", "1e-4", "--out-dir", str(step4_dir)],
        env=JEPA_ENV,
    )
    sections["Step 4: full training on scale50k"] = (
        f"Exit code: {r.returncode}\n\n```\n{r.stdout[-4000:]}\n```"
    )
    if r.returncode != 0:
        sections["Outcome"] = "FAILED at step 4 (full training). Data was generated; retrain manually."
        write_report(sections)
        log("Step 4 failed, stopping.")
        return

    # --- Step 5: final probe ---
    log("Step 5: final probe on step4's best checkpoint")
    r = run(
        [JEPA_PY, "-u", "model/probe.py", "--data-dir", "dataset/scale50k",
         "--checkpoint", str(step4_dir / "checkpoint_best.pt")],
        env=JEPA_ENV,
    )
    final_trained_mse, final_random_mse = parse_probe_output(r.stdout)
    sections["Step 5: final probe on scale50k model"] = (
        f"trained_mse={final_trained_mse} random_mse={final_random_mse}\n\n```\n{r.stdout}\n```"
    )

    sections["Outcome"] = (
        f"Completed full pipeline. 10k-episode gate: trained_mse={trained_mse} "
        f"vs random_mse={random_mse} (passed). Final 50k-episode model: "
        f"trained_mse={final_trained_mse} vs random_mse={final_random_mse}. "
        f"Checkpoints in {step1_dir} and {step4_dir}."
    )
    write_report(sections)
    log("Pipeline complete.")


if __name__ == "__main__":
    main()
