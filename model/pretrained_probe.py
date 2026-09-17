"""Zero-shot control experiment: does Meta's actual pretrained V-JEPA2.1
ViT-B/16 checkpoint (80M params, trained on Kinetics/SSV2/HowTo100M) beat
random-init at decoding agent position on OUR domain, with NO fine-tuning
at all? This disambiguates two hypotheses from docs/phase2-plan.md:
  - if pretrained-frozen ALSO fails to beat random-init: task-design
    problem (masked-video prediction doesn't encode this kind of
    variable, at any scale) -- fine-tuning wouldn't fix it either.
  - if pretrained-frozen DOES beat random-init: scale problem -- our
    from-scratch 2.67M-param/10k-episode setup is undertrained, and both
    fine-tuning and further from-scratch scaling are well-motivated.

Uses RoPE (interpolate_rope=True) so it can run at our native 128x128/
16-frame clip shape instead of their trained 384px/64-frame setup --
their patch_embed/attention weights are resolution-independent; only the
(unused, since use_rope=True) absolute pos_embed would need interpolation.

Requires the facebookresearch/vjepa2 repo cloned at REFS_VJEPA2 below and
the checkpoint downloaded there (see docs/phase2-plan.md for exact URLs).
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

REFS_VJEPA2 = Path("/home/ubuntu/refs/vjepa2")
sys.path.insert(0, str(REFS_VJEPA2))

import torch
from torch.utils.data import DataLoader

from model.attentive_probe import train_and_eval_attentive_probe
from model.dataset import ClipDataset

CHECKPOINT_PATH = "/home/ubuntu/refs/checkpoints/vjepa2_1_vitb_384.pt"


def _clean_backbone_key(state_dict):
    out = {}
    for key, val in state_dict.items():
        key = key.replace("module.", "").replace("backbone.", "")
        out[key] = val
    return out


def build_encoder(img_size: int, num_frames: int, pretrained: bool):
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
    if pretrained:
        sd = torch.load(CHECKPOINT_PATH, map_location="cpu")
        encoder_sd = _clean_backbone_key(sd["ema_encoder"])
        missing, unexpected = encoder.load_state_dict(encoder_sd, strict=False)
        print(f"loaded pretrained weights: {len(missing)} missing, {len(unexpected)} unexpected keys")
        if missing:
            print(f"  missing (first 10): {missing[:10]}")
        if unexpected:
            print(f"  unexpected (first 10): {unexpected[:10]}")
    return encoder


@torch.no_grad()
def embed_all_tokens_pretrained(encoder, loader, device):
    feats, targets = [], []
    for clip, agent_position in loader:
        clip = clip.to(device)  # (B, T, C, H, W)
        clip = clip.permute(0, 2, 1, 3, 4)  # -> (B, C, T, H, W), their expected layout
        out = encoder(clip)  # (B, N, D)
        feats.append(out.cpu())
        targets.append(agent_position)
    return torch.cat(feats, dim=0), torch.cat(targets, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="dataset/pilot")
    parser.add_argument("--clip-len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--probe-steps", type=int, default=500)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = REPO / args.data_dir

    val_ds = ClipDataset(
        data_dir, clip_len=args.clip_len, stride=args.stride, split="val", shuffle=False
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, drop_last=True, num_workers=2)
    print(f"val clips: {len(val_ds)}")

    for name, pretrained in [("vjepa2_pretrained", True), ("vjepa2_random_init", False)]:
        encoder = build_encoder(img_size=128, num_frames=args.clip_len, pretrained=pretrained).to(device)
        encoder.eval()
        feats, targets = embed_all_tokens_pretrained(encoder, val_loader, device)
        mse = train_and_eval_attentive_probe(feats, targets, device, steps=args.probe_steps)
        print(f"{name}: attentive-probe MSE on agent_position = {mse:.4f} (tokens {feats.shape[1]}, dim {feats.shape[2]})")
        del encoder
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
