"""Low-cost real-video demo: does Meta's ACTUAL pretrained V-JEPA2.1
predictor (not our own from-scratch or fine-tuned one) produce a
sensible "imagined future" representation on real video, decoded back to
pixels for a human to look at?

Motivated by the car-nav result: our own fine-tuned predictor, trained for
only a few epochs on a small synthetic corpus, decoded to nothing
recognizable. But the checkpoint at REFS_CHECKPOINT already contains a
*real* predictor -- trained jointly with the encoder on Kinetics/SSV2/
HowTo100M for 40 epochs at world_size=512 -- that we've never actually
used (model/pretrained_probe.py and model/pretrained_jepa.py only ever
loaded the "ema_encoder" key). This script loads both "ema_encoder" and
"predictor" from the same checkpoint file, no training of either, and
runs them on a small real-video sample (dataset/kinetics_mini/, see
scripts/download_kinetics_mini.sh) instead of our synthetic car-nav
renders -- matching the domain the predictor was actually trained on.

The predictor's output lives in a 1664-dim "teacher" space, not the
encoder's own 768-dim space (see src/hub/backbones.py's
vjepa2_1_teacher_embed_dim=1664 -- V-JEPA2.1 distills all model sizes'
predictors into one shared teacher space). So this script trains two
small FrameDecoders, not one: one from the encoder's raw 768-dim output
(sanity check: does the encoder's own representation contain enough
information to reconstruct a frame at all), and one from the predictor's
1664-dim predicted-token output (the actual test: does the *imagined*
representation decode to something resembling the true future frame).
Both decoders are cheap -- only their own small conv params get
gradients, both real-V-JEPA2 modules stay frozen throughout.

v2 note: the first version masked the entire last temporal tubelet as one
solid block (predict-the-whole-future-frame). That's out-of-distribution
for how this predictor was actually trained: V-JEPA2.1's own mask config
(configs/train_2_1/vitb16/pretrain-256px-16f.yaml) uses several small
scattered spatial blocks broadcast across the FULL clip duration (temporal_
scale=1.0), i.e. "fill in masked patches given visible patches at every
timestep", not "predict a future timestep from past ones". This version
uses model.jepa.MultiBlockMask (same scattered-block scheme) instead, and
reconstructs the last frame's token grid by combining the predictor's
teacher-space output for masked cells with the SAME predictor's
teacher-space projection of context cells (predictor_proj_context, i.e.
`x_context`, previously discarded) for visible cells -- both live in the
1664-dim teacher space, unlike the raw 768-dim encoder output, so they
can't be mixed with encoder tokens directly.
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
REFS_VJEPA2 = Path("/home/ubuntu/refs/vjepa2")
sys.path.insert(0, str(REFS_VJEPA2))

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.decoder import FrameDecoder, last_tubelet_tokens
from model.jepa import JepaConfig, MultiBlockMask, split_mask_indices
from model.real_video_dataset import RealVideoClipDataset

CHECKPOINT_PATH = "/home/ubuntu/refs/checkpoints/vjepa2_1_vitb_384.pt"
TEACHER_EMBED_DIM = 1664


def _clean_backbone_key(state_dict):
    return {k.replace("module.", "").replace("backbone.", ""): v for k, v in state_dict.items()}


def build_encoder(img_size: int, num_frames: int):
    from app.vjepa_2_1.models.vision_transformer import vit_base

    encoder = vit_base(
        patch_size=16,
        img_size=(img_size, img_size),
        num_frames=num_frames,
        tubelet_size=2,
        use_sdpa=True,
        use_silu=False,
        wide_silu=True,
        uniform_power=False,
        use_rope=True,
        img_temporal_dim_size=1,
        interpolate_rope=True,
    )
    sd = torch.load(CHECKPOINT_PATH, map_location="cpu")
    missing, unexpected = encoder.load_state_dict(_clean_backbone_key(sd["ema_encoder"]), strict=False)
    print(f"encoder: {len(missing)} missing, {len(unexpected)} unexpected keys")
    return encoder


def build_predictor(img_size: int, num_frames: int):
    from app.vjepa_2_1.models.predictor import vit_predictor

    predictor = vit_predictor(
        img_size=(img_size, img_size),
        patch_size=16,
        use_mask_tokens=True,
        embed_dim=768,
        predictor_embed_dim=384,
        teacher_embed_dim=TEACHER_EMBED_DIM,
        num_frames=num_frames,
        tubelet_size=2,
        depth=12,
        num_heads=12,
        num_mask_tokens=8,
        use_rope=True,
        uniform_power=False,
        use_sdpa=True,
        use_silu=False,
        wide_silu=True,
        n_output_distillation=1,
        return_all_tokens=True,
        img_temporal_dim_size=1,
        interpolate_rope=True,
    )
    sd = torch.load(CHECKPOINT_PATH, map_location="cpu")
    missing, unexpected = predictor.load_state_dict(_clean_backbone_key(sd["predictor"]), strict=False)
    print(f"predictor: {len(missing)} missing, {len(unexpected)} unexpected keys")
    return predictor


def train_decoder(decoder, embed_fn, loader, device, epochs, lr, tag, best_path=None):
    opt = torch.optim.Adam(decoder.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)
    best = float("inf")
    for epoch in range(epochs):
        decoder.train()
        total, n = 0.0, 0
        for clip, _ in loader:
            clip = clip.to(device)
            target = clip[:, -1]
            with torch.no_grad():
                tokens = embed_fn(clip)
            pred = decoder(tokens)
            loss = nn.functional.mse_loss(pred, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            n += 1
        sched.step()
        avg = total / max(1, n)
        marker = ""
        if avg < best:
            best = avg
            marker = " *"
            if best_path is not None:
                torch.save({"decoder": decoder.state_dict()}, best_path)
        print(f"[{tag}] epoch {epoch:03d} loss={avg:.5f} lr={sched.get_last_lr()[0]:.2e}{marker}")
    if best_path is not None:
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        decoder.load_state_dict(ckpt["decoder"])
        print(f"[{tag}] restored best checkpoint (loss={best:.5f}) from {best_path}")
    return decoder


def reconstruct_last_tubelet(ctx_out, pred_out, visible_idx, masked_idx, n_temporal, g2, dim):
    """Combine the predictor's teacher-space output for the last tubelet's
    masked spatial cells (pred_out) with its teacher-space projection of
    the same tubelet's visible cells (ctx_out) into one full (B, g2, dim)
    grid -- the token-level analogue of "fill in the masked patches, keep
    the visible ones", matching how V-JEPA2 was actually trained (unlike
    v1, which discarded ctx_out and predicted the whole frame from scratch)."""
    B = ctx_out.shape[0]
    last_start = (n_temporal - 1) * g2
    out = torch.zeros(B, g2, dim, device=ctx_out.device, dtype=ctx_out.dtype)
    for b in range(B):
        vis_mask = (visible_idx[b] >= last_start) & (visible_idx[b] < last_start + g2)
        v_cells = visible_idx[b][vis_mask] - last_start
        out[b, v_cells] = ctx_out[b, vis_mask.nonzero(as_tuple=True)[0]]

        msk_mask = (masked_idx[b] >= last_start) & (masked_idx[b] < last_start + g2)
        m_cells = masked_idx[b][msk_mask] - last_start
        out[b, m_cells] = pred_out[b, msk_mask.nonzero(as_tuple=True)[0]]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="dataset/kinetics_mini")
    parser.add_argument("--clip-len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-samples", type=int, default=6)
    parser.add_argument("--mask-ratio", type=float, default=0.4)
    parser.add_argument("--n-blocks", type=int, default=6)
    parser.add_argument("--out-path", type=str, default="notes/figures/prediction_demo_realvideo.png")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = JepaConfig()  # img_size=128, patch_size=16, tubelet_size=2, clip_len=16 -> matches our clip shape
    cfg.mask_ratio = args.mask_ratio
    masker = MultiBlockMask(cfg, n_blocks=args.n_blocks)
    data_dir = REPO / args.data_dir

    encoder = build_encoder(cfg.img_size, args.clip_len).to(device).eval()
    predictor = build_predictor(cfg.img_size, args.clip_len).to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    for p in predictor.parameters():
        p.requires_grad_(False)

    train_ds = RealVideoClipDataset(data_dir, split="train", clip_len=args.clip_len, stride=args.stride, img_size=cfg.img_size)
    val_ds = RealVideoClipDataset(data_dir, split="val", clip_len=args.clip_len, stride=args.stride, img_size=cfg.img_size, shuffle=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, drop_last=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.n_samples, drop_last=True, num_workers=2)

    encoder_dim = 768  # ViT-B/16 -- cfg.encoder_dim defaults to 192 (our from-scratch model), not used here
    g2 = cfg.grid_size * cfg.grid_size

    def embed_true(clip):
        clip_v = clip.permute(0, 2, 1, 3, 4)
        return last_tubelet_tokens(encoder(clip_v), cfg.grid_size)

    def embed_pred(clip):
        b = clip.shape[0]
        mask = masker.sample(b, clip.device)  # fresh scattered blocks every batch
        visible_idx, masked_idx = split_mask_indices(mask)
        clip_v = clip.permute(0, 2, 1, 3, 4)
        context_out = encoder(clip_v, masks=[visible_idx])
        pred_out, ctx_out = predictor(context_out, [visible_idx], [masked_idx], mod="video", mask_index=1)
        return reconstruct_last_tubelet(ctx_out, pred_out, visible_idx, masked_idx, cfg.n_temporal_tokens, g2, TEACHER_EMBED_DIM)

    decoder_true = FrameDecoder(embed_dim=encoder_dim, grid_size=cfg.grid_size, img_size=cfg.img_size).to(device)
    decoder_pred = FrameDecoder(embed_dim=TEACHER_EMBED_DIM, grid_size=cfg.grid_size, img_size=cfg.img_size).to(device)

    ckpt_dir = REPO / "runs" / "demo_realvideo_decoders"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("training decoder_true (encoder's own raw representation)...")
    train_decoder(decoder_true, embed_true, train_loader, device, args.epochs, args.lr, "true", best_path=ckpt_dir / "decoder_true_best.pt")
    print("training decoder_pred (predictor's imagined-future representation, scattered mask)...")
    train_decoder(decoder_pred, embed_pred, train_loader, device, args.epochs, args.lr, "pred", best_path=ckpt_dir / "decoder_pred_best.pt")

    decoder_true.eval()
    decoder_pred.eval()
    clip, _ = next(iter(val_loader))
    clip = clip.to(device)

    with torch.no_grad():
        actual_frame = clip[:, -1]
        clip_v = clip.permute(0, 2, 1, 3, 4)
        true_tokens = last_tubelet_tokens(encoder(clip_v), cfg.grid_size)
        decoded_true = decoder_true(true_tokens)

        mask_v = masker.sample(clip.shape[0], clip.device)
        visible_idx_v, masked_idx_v = split_mask_indices(mask_v)
        context_out = encoder(clip_v, masks=[visible_idx_v])
        pred_out, ctx_out = predictor(context_out, [visible_idx_v], [masked_idx_v], mod="video", mask_index=1)
        combined = reconstruct_last_tubelet(ctx_out, pred_out, visible_idx_v, masked_idx_v, cfg.n_temporal_tokens, g2, TEACHER_EMBED_DIM)
        decoded_pred = decoder_pred(combined)

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
    out_path = REPO / args.out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
