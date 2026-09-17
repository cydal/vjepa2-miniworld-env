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
from model.jepa import JEPA, JepaConfig, ema_momentum_schedule


def cosine_warmup_lr(step: int, total_steps: int, warmup_steps: int, base_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


DIAG_KEYS = ("target_std", "context_std", "target_norm_weight", "context_norm_weight")


def evaluate(model, loader, device):
    model.eval()
    total_loss, totals, n = 0.0, {k: 0.0 for k in DIAG_KEYS}, 0
    with torch.no_grad():
        for clip, _ in loader:
            clip = clip.to(device)
            loss, stats = model(clip)
            total_loss += loss.item()
            for k in DIAG_KEYS:
                totals[k] += stats[k]
            n += 1
    model.train()
    n = max(1, n)
    return total_loss / n, {k: v / n for k, v in totals.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="dataset/pilot")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--clip-len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--mask-ratio", type=float, default=None)
    parser.add_argument("--ema-momentum-start", type=float, default=None)
    parser.add_argument("--ema-momentum-end", type=float, default=None)
    parser.add_argument("--clip-grad", type=float, default=10.0)
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
        data_dir, clip_len=args.clip_len, stride=args.stride, split="train",
        shuffle=True, chunk_size=args.chunk_size,
    )
    val_ds = ClipDataset(
        data_dir, clip_len=args.clip_len, stride=args.stride, split="val",
        shuffle=False, chunk_size=args.chunk_size,
    )
    print(f"train clips: {len(train_ds)}, val clips: {len(val_ds)}")

    # shuffling happens inside ClipDataset (block shuffle over episode
    # chunks, to keep peak memory bounded) -- DataLoader must not be asked
    # to shuffle an IterableDataset itself. num_workers>0 parallelizes mp4
    # decode (the actual bottleneck at this corpus size) across this box's
    # 4 CPU cores instead of decoding single-threaded. persistent_workers
    # avoids re-forking workers every epoch; prefetch_factor keeps several
    # batches decoded ahead of the GPU so it isn't stalling on I/O -- this
    # model is tiny (a few MB of activations per batch) so decode, not GPU
    # compute, is the default bottleneck unless batches are kept flowing.
    loader_kwargs = dict(
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, drop_last=True, **loader_kwargs)

    cfg_overrides = {}
    if args.mask_ratio is not None:
        cfg_overrides["mask_ratio"] = args.mask_ratio
    if args.ema_momentum_start is not None:
        cfg_overrides["ema_momentum_start"] = args.ema_momentum_start
    if args.ema_momentum_end is not None:
        cfg_overrides["ema_momentum_end"] = args.ema_momentum_end
    cfg = JepaConfig(clip_len=args.clip_len, **cfg_overrides)
    print(f"mask_ratio={cfg.mask_ratio} ema_momentum={cfg.ema_momentum_start}->{cfg.ema_momentum_end} clip_grad={args.clip_grad}")
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
            epoch_loss, epoch_totals = 0.0, {k: 0.0 for k in DIAG_KEYS}
            for clip, _ in train_loader:
                clip = clip.to(device)
                lr = cosine_warmup_lr(step, total_steps, warmup_steps, args.lr)
                for g in opt.param_groups:
                    g["lr"] = lr
                momentum = ema_momentum_schedule(
                    step, total_steps, cfg.ema_momentum_start, cfg.ema_momentum_end
                )

                loss, stats = model(clip)
                opt.zero_grad()
                loss.backward()
                # V-JEPA1 clips encoder/predictor grad norms separately,
                # only after warmup (app/vjepa/train.py) -- cheap safety
                # net against gradient spikes; dropped in V-JEPA2 (which
                # relies on the target-side LayerNorm instead) but harmless
                # to keep here too.
                if step >= warmup_steps and args.clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(model.context_encoder.parameters(), args.clip_grad)
                    torch.nn.utils.clip_grad_norm_(model.predictor.parameters(), args.clip_grad)
                opt.step()
                model.update_target_encoder(momentum=momentum)

                epoch_loss += loss.item()
                for k in DIAG_KEYS:
                    epoch_totals[k] += stats[k]
                step += 1

            n_batches = max(1, len(train_loader))
            train_loss = epoch_loss / n_batches
            train_diag = {k: v / n_batches for k, v in epoch_totals.items()}
            val_loss, val_diag = evaluate(model, val_loader, device)

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                **{f"train_{k}": v for k, v in train_diag.items()},
                **{f"val_{k}": v for k, v in val_diag.items()},
                "lr": lr,
                "momentum": momentum,
                "elapsed_s": time.time() - t0,
            }
            log_f.write(json.dumps(row) + "\n")
            log_f.flush()
            print(
                f"epoch {epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"train_std={train_diag['target_std']:.4f} val_std={val_diag['target_std']:.4f} "
                f"ctx_std={train_diag['context_std']:.4f} "
                f"tgt_norm_w={train_diag['target_norm_weight']:.4f} "
                f"ctx_norm_w={train_diag['context_norm_weight']:.4f} "
                f"lr={lr:.2e} m={momentum:.5f} ({row['elapsed_s']:.1f}s)"
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
