"""Small sanity-run trainer: plain V-JEPA-style pretraining on the MiniGrid
pilot corpus. Not a serious first pretraining pass -- just enough signal
(loss curve, collapse diagnostic) to decide whether scaling up data
generation is worth it. See docs/phase2-plan.md.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from model.dataset import ClipDataset
from model.jepa import JEPA, JepaConfig


def cosine_warmup_lr(step: int, total_steps: int, warmup_steps: int, base_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def evaluate(model, loader, device):
    model.eval()
    total_loss, total_std, n = 0.0, 0.0, 0
    with torch.no_grad():
        for clip, _ in loader:
            clip = clip.to(device)
            loss, stats = model(clip)
            total_loss += loss.item()
            total_std += stats["target_std"]
            n += 1
    model.train()
    return total_loss / max(1, n), total_std / max(1, n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="dataset/pilot")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--clip-len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    data_dir = Path(__file__).resolve().parent.parent / args.data_dir
    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else (
        Path(__file__).resolve().parent.parent / "runs" / run_id
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    train_ds = ClipDataset(
        data_dir, clip_len=args.clip_len, stride=args.stride, split="train", shuffle=True
    )
    val_ds = ClipDataset(
        data_dir, clip_len=args.clip_len, stride=args.stride, split="val", shuffle=False
    )
    print(f"train clips: {len(train_ds)}, val clips: {len(val_ds)}")

    # shuffling happens inside ClipDataset (block shuffle over episode
    # chunks, to keep peak memory bounded) -- DataLoader must not be asked
    # to shuffle an IterableDataset itself. num_workers>0 parallelizes mp4
    # decode (the actual bottleneck at this corpus size) across this box's
    # 4 CPU cores instead of decoding single-threaded.
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, drop_last=True, num_workers=3)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, drop_last=True, num_workers=2)

    cfg = JepaConfig(clip_len=args.clip_len)
    model = JEPA(cfg).to(device)
    n_params = sum(p.numel() for p in model.context_encoder.parameters())
    print(f"context encoder params: {n_params / 1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.04)
    total_steps = args.epochs * max(1, len(train_loader))
    warmup_steps = max(1, total_steps // 10)

    log_path = out_dir / "log.jsonl"
    best_val_loss = float("inf")
    best_ckpt_path = out_dir / "checkpoint_best.pt"
    step = 0
    with open(log_path, "w") as log_f:
        for epoch in range(args.epochs):
            t0 = time.time()
            epoch_loss, epoch_std = 0.0, 0.0
            for clip, _ in train_loader:
                clip = clip.to(device)
                lr = cosine_warmup_lr(step, total_steps, warmup_steps, args.lr)
                for g in opt.param_groups:
                    g["lr"] = lr

                loss, stats = model(clip)
                opt.zero_grad()
                loss.backward()
                opt.step()
                model.update_target_encoder()

                epoch_loss += loss.item()
                epoch_std += stats["target_std"]
                step += 1

            n_batches = max(1, len(train_loader))
            train_loss = epoch_loss / n_batches
            train_std = epoch_std / n_batches
            val_loss, val_std = evaluate(model, val_loader, device)

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_target_std": train_std,
                "val_target_std": val_std,
                "lr": lr,
                "elapsed_s": time.time() - t0,
            }
            log_f.write(json.dumps(row) + "\n")
            log_f.flush()
            print(
                f"epoch {epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"train_std={train_std:.4f} val_std={val_std:.4f} lr={lr:.2e} "
                f"({row['elapsed_s']:.1f}s)"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {"model": model.state_dict(), "cfg": cfg.__dict__, "epoch": epoch},
                    best_ckpt_path,
                )

    ckpt_path = out_dir / "checkpoint_final.pt"
    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, ckpt_path)
    print(f"saved final checkpoint to {ckpt_path}")
    print(f"saved best checkpoint (val_loss={best_val_loss:.4f}) to {best_ckpt_path}")
    print(f"run dir: {out_dir}")


if __name__ == "__main__":
    main()
