"""JEPA training wrapper around Meta's pretrained V-JEPA2.1 ViT-B/16
encoder, for fine-tuning on our MiniGrid domain instead of training an
encoder from scratch. Motivated by model/pretrained_probe.py's result:
the pretrained encoder beats random-init at decoding agent position
zero-shot (no fine-tuning at all) -- our from-scratch 2.67M-param encoder
does not. See docs/phase2-plan.md.

Reuses the masking/predictor/EMA machinery from model/jepa.py unchanged;
only the encoder itself is swapped for the real V-JEPA2.1 VisionTransformer
(RoPE-based, so it runs natively at our 128px/16-frame clip shape --
verified in pretrained_probe.py, 0 missing/unexpected keys on load).
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
REFS_VJEPA2 = Path("/home/ubuntu/refs/vjepa2")
sys.path.insert(0, str(REFS_VJEPA2))

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.jepa import (
    JepaConfig,
    MultiBlockMask,
    Predictor,
    batched_gather,
    split_mask_indices,
)

CHECKPOINT_PATH = "/home/ubuntu/refs/checkpoints/vjepa2_1_vitb_384.pt"
VJEPA2_ENCODER_DIM = 768


def _clean_backbone_key(state_dict):
    out = {}
    for key, val in state_dict.items():
        key = key.replace("module.", "").replace("backbone.", "")
        out[key] = val
    return out


def build_pretrained_encoder(img_size: int, num_frames: int):
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
    encoder.load_state_dict(_clean_backbone_key(sd["ema_encoder"]), strict=False)
    return encoder


class PretrainedJEPA(nn.Module):
    """Same forward()/update_target_encoder() interface as model.jepa.JEPA,
    so it's a drop-in swap in a training loop -- only the encoder
    construction differs (pretrained ViT-B/16 instead of a from-scratch
    small ViT, no separate patch_embed/pos_embed since their encoder
    handles that internally)."""

    def __init__(self, cfg: JepaConfig):
        super().__init__()
        cfg.encoder_dim = VJEPA2_ENCODER_DIM  # Predictor.in_proj must match
        self.cfg = cfg
        self.context_encoder = build_pretrained_encoder(cfg.img_size, cfg.clip_len)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.predictor = Predictor(cfg)
        self.masker = MultiBlockMask(cfg)

    @torch.no_grad()
    def update_target_encoder(self, momentum: float = None):
        m = self.cfg.ema_momentum_start if momentum is None else momentum
        for tp, cp in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            tp.mul_(m).add_(cp, alpha=1 - m)

    def forward(self, clip: torch.Tensor):
        """clip: (B, T, C, H, W) in [0, 1] -- our layout; V-JEPA2 expects
        (B, C, T, H, W)."""
        b, n_tokens = clip.shape[0], self.cfg.n_tokens
        clip_v = clip.permute(0, 2, 1, 3, 4)

        mask = self.masker.sample(b, clip.device)
        visible_idx, masked_idx = split_mask_indices(mask)

        with torch.no_grad():
            target_out = self.target_encoder(clip_v)  # (B, n_tokens, 768), full clip
            target_out = F.layer_norm(target_out, (target_out.shape[-1],))
            target_masked = batched_gather(target_out, masked_idx)

        context_out = self.context_encoder(clip_v, masks=[visible_idx])  # (B, n_visible, 768)
        pred_masked = self.predictor(context_out, visible_idx, masked_idx, n_tokens)

        loss = F.l1_loss(pred_masked, target_masked.detach())
        stats = {
            "target_std": target_out.detach().std(dim=(0, 1)).mean().item(),
            "context_std": context_out.detach().std(dim=(0, 1)).mean().item(),
            "n_masked": mask[0].sum().item(),
            "n_visible": (~mask[0]).sum().item(),
        }
        return loss, stats
