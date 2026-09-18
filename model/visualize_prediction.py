"""The actual payoff demo: does the predictor's imagined future representation
correspond to something sensible, visually? For each sample clip, masks
only the *last* temporal tubelet (not a random block, unlike training --
this makes the demo legible: "predict the last frame from everything
before it"), runs it through the predictor, and decodes three things
side by side:

  actual frame | decoded from the TRUE encoder representation | decoded from the PREDICTOR's imagined representation

The middle panel is a sanity check on the decoder itself (does it
faithfully reconstruct from a real embedding at all); the right panel is
the real test (model/decoder.py + model/train_decoder.py must both exist
and the decoder must already be trained). Not part of any automatic
pipeline -- run manually.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from model.dataset import ClipDataset
from model.decoder import FrameDecoder, last_tubelet_tokens
from model.jepa import JepaConfig, split_mask_indices
from model.pretrained_jepa import PretrainedJEPA


def last_tubelet_mask(batch_size: int, cfg: JepaConfig, device) -> torch.Tensor:
    """(B, n_tokens) bool, True only for the last temporal tubelet's spatial
    grid -- a deterministic, interpretable mask instead of MultiBlockMask's
    random blocks, so the demo reads as "predict the future from the past."
    """
    g2 = cfg.grid_size * cfg.grid_size
    mask = torch.zeros(batch_size, cfg.n_tokens, dtype=torch.bool, device=device)
    mask[:, -g2:] = True
    return mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--jepa-checkpoint", type=str, default=None,
                         help="fine-tuned PretrainedJEPA checkpoint; omit to use the zero-shot pretrained encoder as-is")
    parser.add_argument("--decoder-checkpoint", type=str, required=True)
    parser.add_argument("--clip-len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--n-samples", type=int, default=6)
    parser.add_argument("--out-path", type=str, default="notes/figures/prediction_demo.png")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent.parent
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = JepaConfig()
    jepa = PretrainedJEPA(cfg).to(device)
    ckpt = torch.load(args.jepa_checkpoint, map_location=device, weights_only=False)
    jepa.load_state_dict(ckpt["model"])
    jepa.eval()

    decoder = FrameDecoder(embed_dim=cfg.encoder_dim, grid_size=cfg.grid_size, img_size=cfg.img_size).to(device)
    dckpt = torch.load(args.decoder_checkpoint, map_location=device, weights_only=False)
    decoder.load_state_dict(dckpt["decoder"])
    decoder.eval()

    val_ds = ClipDataset(repo / args.data_dir, clip_len=args.clip_len, stride=args.stride, split="val", shuffle=True)
    loader = DataLoader(val_ds, batch_size=args.n_samples, drop_last=True, num_workers=2)
    clip, _ = next(iter(loader))
    clip = clip.to(device)  # (B, T, C, H, W)
    clip_v = clip.permute(0, 2, 1, 3, 4)

    with torch.no_grad():
        actual_frame = clip[:, -1]  # (B, C, H, W), ground truth pixels

        true_tokens = jepa.context_encoder(clip_v)
        true_last = last_tubelet_tokens(true_tokens, cfg.grid_size)
        decoded_true = decoder(true_last)

        mask = last_tubelet_mask(clip.shape[0], cfg, device)
        visible_idx, masked_idx = split_mask_indices(mask)
        # PretrainedJEPA's encoder takes a mask list directly (its own
        # patchify + apply_masks, matching how PretrainedJEPA.forward() uses it)
        # rather than JEPA's separate patch_embed-then-gather.
        context_out = jepa.context_encoder(clip_v, masks=[visible_idx])
        pred_masked = jepa.predictor(context_out, visible_idx, masked_idx, cfg.n_tokens)
        decoded_pred = decoder(pred_masked)

    n = clip.shape[0]
    fig, axes = plt.subplots(n, 3, figsize=(6, 2 * n))
    col_titles = ["actual frame", "decoded (true embedding)", "decoded (predicted embedding)"]
    for row in range(n):
        for col, img in enumerate([actual_frame[row], decoded_true[row], decoded_pred[row]]):
            arr = img.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            axes[row, col].imshow(arr)
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(col_titles[col], fontsize=9)
    fig.tight_layout()
    out_path = repo / args.out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
