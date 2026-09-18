"""Evaluate a fine-tuned PretrainedJEPA checkpoint (model/finetune.py's
output) with the attentive probe, against a fresh random-init encoder of
the same architecture -- the same comparison used throughout this session,
now as a reusable script instead of an inline snippet.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from model.attentive_probe import train_and_eval_attentive_probe
from model.dataset import ClipDataset
from model.jepa import JepaConfig
from model.pretrained_jepa import PretrainedJEPA


@torch.no_grad()
def embed_all_tokens(encoder, loader, device, max_clips):
    feats, targets, n = [], [], 0
    for clip, agent_position in loader:
        clip = clip.to(device).permute(0, 2, 1, 3, 4)
        out = encoder(clip)
        feats.append(out.cpu())
        targets.append(agent_position)
        n += clip.shape[0]
        if n >= max_clips:
            break
    return torch.cat(feats, dim=0), torch.cat(targets, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--clip-len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-clips", type=int, default=1500)
    parser.add_argument("--probe-steps", type=int, default=500)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = Path(__file__).resolve().parent.parent / args.data_dir

    val_ds = ClipDataset(data_dir, clip_len=args.clip_len, stride=args.stride, split="val", shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, drop_last=True, num_workers=2)
    print(f"val clips: {len(val_ds)}")

    cfg = JepaConfig()
    finetuned = PretrainedJEPA(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    finetuned.load_state_dict(ckpt["model"])
    finetuned.eval()

    # NOTE: PretrainedJEPA.__init__ always loads Meta's pretrained weights
    # (there's no random-init path in that class) -- this is a second
    # zero-shot-pretrained measurement, NOT a true random-init baseline.
    # Use model/pretrained_probe.py for a real random-init comparison.
    zero_shot = PretrainedJEPA(cfg).to(device)
    zero_shot.eval()

    for name, model in [("finetuned", finetuned), ("zero_shot_pretrained", zero_shot)]:
        feats, targets = embed_all_tokens(model.context_encoder, val_loader, device, args.max_clips)
        mse = train_and_eval_attentive_probe(feats, targets, device, steps=args.probe_steps)
        print(f"{name}: attentive-probe MSE on agent_position = {mse:.4f} (tokens {feats.shape[1]}, dim {feats.shape[2]})")


if __name__ == "__main__":
    main()
