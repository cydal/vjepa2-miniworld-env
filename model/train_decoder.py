"""Train the visualization decoder against a frozen, already-fine-tuned
encoder's real (non-predicted) representations -- teaches the decoder to
map "true" embeddings back to pixels faithfully, before ever looking at a
predicted embedding. The actual point (predicted representation -> decoded
image, compared against the real future frame) is model/visualize_prediction.py,
which reuses this trained decoder.

Cheap relative to the encoder fine-tune: only decoder.parameters() get
gradients, the 87M-param encoder stays frozen and in eval() the whole time.
Not part of the automatic overnight pipeline -- run manually once wanted.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.dataset import ClipDataset
from model.decoder import FrameDecoder, last_tubelet_tokens
from model.jepa import JepaConfig
from model.pretrained_jepa import PretrainedJEPA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--encoder-checkpoint", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--clip-len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent.parent
    data_dir = repo / args.data_dir
    out_dir = Path(args.out_dir) if args.out_dir else repo / "runs" / f"decoder-{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = JepaConfig()
    jepa = PretrainedJEPA(cfg).to(device)
    ckpt = torch.load(args.encoder_checkpoint, map_location=device, weights_only=False)
    jepa.load_state_dict(ckpt["model"])
    encoder = jepa.context_encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    decoder = FrameDecoder(embed_dim=cfg.encoder_dim, grid_size=cfg.grid_size, img_size=cfg.img_size).to(device)
    opt = torch.optim.Adam(decoder.parameters(), lr=args.lr)

    train_ds = ClipDataset(data_dir, clip_len=args.clip_len, stride=args.stride, split="train", shuffle=True)
    val_ds = ClipDataset(data_dir, clip_len=args.clip_len, stride=args.stride, split="val", shuffle=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, drop_last=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, drop_last=True, num_workers=args.num_workers)
    print(f"train clips: {len(train_ds)}, val clips: {len(val_ds)}")

    best_val = float("inf")
    for epoch in range(args.epochs):
        t0 = time.time()
        decoder.train()
        train_loss = 0.0
        n_batches = 0
        for clip, _ in train_loader:
            clip = clip.to(device)  # (B, T, C, H, W) in [0, 1]
            target = clip[:, -1]  # last raw frame, (B, C, H, W)
            with torch.no_grad():
                tokens = encoder(clip.permute(0, 2, 1, 3, 4))
                last_tokens = last_tubelet_tokens(tokens, cfg.grid_size)
            pred = decoder(last_tokens)
            loss = nn.functional.mse_loss(pred, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()
            n_batches += 1
        train_loss /= max(1, n_batches)

        decoder.eval()
        val_loss, n_val = 0.0, 0
        with torch.no_grad():
            for clip, _ in val_loader:
                clip = clip.to(device)
                target = clip[:, -1]
                tokens = encoder(clip.permute(0, 2, 1, 3, 4))
                last_tokens = last_tubelet_tokens(tokens, cfg.grid_size)
                pred = decoder(last_tokens)
                val_loss += nn.functional.mse_loss(pred, target).item()
                n_val += 1
        val_loss /= max(1, n_val)

        print(f"epoch {epoch:03d} train_loss={train_loss:.5f} val_loss={val_loss:.5f} ({time.time()-t0:.1f}s)")
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"decoder": decoder.state_dict(), "cfg": cfg.__dict__}, out_dir / "decoder_best.pt")

    print(f"best val_loss={best_val:.5f}, saved to {out_dir / 'decoder_best.pt'}")


if __name__ == "__main__":
    main()
