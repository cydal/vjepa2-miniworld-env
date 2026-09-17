"""Unattended overnight run: fine-tune the pretrained V-JEPA2.1 encoder on
the car-nav pilot dataset, then run the attentive probe automatically on
completion. Self-contained and read-only from the outside -- no process
needs to be inspected or touched while this runs (a manual `kill` on a
DataLoader worker mid-run is exactly what crashed the previous attempt).

Deliberately does NOT chain into anything heavier afterward (e.g. AC
training) -- per explicit instruction tonight, nothing that could risk
system stability while unattended. Next steps after this are a deliberate,
supervised decision, not an automatic one.
"""
import json
import os
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
JEPA_PY = "/home/ubuntu/miniconda3/envs/jepa/bin/python"
LOG_PATH = REPO / "overnight_carnav_log.txt"
REPORT_PATH = REPO / "overnight_carnav_report.md"

JEPA_ENV = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}


def log(msg: str):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def run(cmd, env=None):
    log(f"RUN: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
    log(f"exit={result.returncode}")
    if result.stdout:
        log("stdout tail:\n" + "\n".join(result.stdout.splitlines()[-20:]))
    if result.returncode != 0 and result.stderr:
        log("stderr tail:\n" + "\n".join(result.stderr.splitlines()[-30:]))
    return result


def parse_probe_output(stdout: str):
    trained_mse, random_mse = None, None
    for line in stdout.splitlines():
        if line.startswith("trained:"):
            trained_mse = float(line.split("=")[1].split("(")[0].strip())
        elif line.startswith("random_init:"):
            random_mse = float(line.split("=")[1].split("(")[0].strip())
    return trained_mse, random_mse


def write_report(sections: dict):
    with open(REPORT_PATH, "w") as f:
        f.write("# Overnight car-nav fine-tune report\n\n")
        for title, body in sections.items():
            f.write(f"## {title}\n\n{body}\n\n")


def main():
    sections = {}
    log("=" * 60)
    log("overnight_carnav_finetune.py starting")

    out_dir = REPO / "runs" / "finetune_carnav_overnight"
    out_dir.mkdir(parents=True, exist_ok=True)

    log("Step 1: fine-tune on dataset/carnav_pilot (10 epochs, lr=2e-5, encoder-lr-mult=0.05, "
        "num-workers=2 for extra RAM headroom overnight)")
    r = run(
        [JEPA_PY, "-u", "model/finetune.py",
         "--data-dir", "dataset/carnav_pilot",
         "--epochs", "10", "--batch-size", "32", "--lr", "2e-5",
         "--encoder-lr-mult", "0.05", "--num-workers", "2",
         "--out-dir", str(out_dir)],
        env=JEPA_ENV,
    )
    sections["Step 1: fine-tune"] = f"Exit code: {r.returncode}\n\n```\n{r.stdout[-4000:]}\n```"
    if r.returncode != 0:
        sections["Outcome"] = "FAILED at fine-tuning step. See stdout above / overnight_carnav_log.txt."
        write_report(sections)
        log("Fine-tuning failed, stopping.")
        return

    log("Step 2: attentive probe on the fine-tuned checkpoint vs a fresh random-init encoder")
    ckpt = out_dir / "checkpoint_best.pt"
    r = run(
        [JEPA_PY, "-u", "model/eval_carnav_finetune.py",
         "--data-dir", "dataset/carnav_pilot", "--checkpoint", str(ckpt),
         "--max-clips", "1500"],
        env=JEPA_ENV,
    )
    sections["Step 2: attentive-probe eval"] = f"Exit code: {r.returncode}\n\n```\n{r.stdout[-3000:]}\n```"

    sections["Outcome"] = f"Fine-tuning completed. Checkpoint at {ckpt}. See Step 2 for the probe comparison."
    write_report(sections)
    log("Pipeline complete.")


if __name__ == "__main__":
    main()
