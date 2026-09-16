"""Plain V-JEPA-style spatiotemporal masked-prediction model: a small ViT
context encoder, an EMA target encoder (stop-gradient), and a shallow
predictor that guesses the target encoder's representation at masked
tubelets from the context encoder's representation at visible ones. No
action conditioning here -- that's a later phase (see docs/phase2-plan.md).
"""
import copy
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class JepaConfig:
    img_size: int = 128
    patch_size: int = 16
    tubelet_size: int = 2
    clip_len: int = 16
    in_channels: int = 3
    encoder_dim: int = 192
    encoder_depth: int = 6
    encoder_heads: int = 6
    predictor_dim: int = 128
    predictor_depth: int = 4
    predictor_heads: int = 4
    mlp_ratio: float = 4.0
    mask_ratio: float = 0.6
    ema_momentum: float = 0.998

    @property
    def grid_size(self) -> int:
        return self.img_size // self.patch_size

    @property
    def n_temporal_tokens(self) -> int:
        return self.clip_len // self.tubelet_size

    @property
    def n_tokens(self) -> int:
        return self.n_temporal_tokens * self.grid_size * self.grid_size


class PatchEmbed3D(nn.Module):
    """Tubelet embedding: (B,T,C,H,W) -> (B, n_tokens, embed_dim)."""

    def __init__(self, cfg: JepaConfig, embed_dim: int):
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Conv3d(
            cfg.in_channels,
            embed_dim,
            kernel_size=(cfg.tubelet_size, cfg.patch_size, cfg.patch_size),
            stride=(cfg.tubelet_size, cfg.patch_size, cfg.patch_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W) -> conv3d wants (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        x = self.proj(x)  # (B, embed_dim, T', H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, n_tokens, embed_dim)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    """Encodes a set of tokens (with their positional embeddings already
    added by the caller) through a stack of transformer blocks."""

    def __init__(self, dim: int, depth: int, heads: int, mlp_ratio: float):
        super().__init__()
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, heads, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


def sincos_pos_embed(n_temporal: int, grid_size: int, dim: int) -> torch.Tensor:
    """Fixed 3D sin-cos positional embedding, split dim into temporal/H/W
    thirds (rounded so they sum back to dim)."""

    def _1d(n: int, d: int) -> torch.Tensor:
        d = d - (d % 2)
        pos = torch.arange(n, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(10000.0) / d)
        )
        pe = torch.zeros(n, d)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe

    d_t = dim // 3
    d_h = dim // 3
    d_w = dim - d_t - d_h
    pe_t = _1d(n_temporal, d_t)
    pe_h = _1d(grid_size, d_h)
    pe_w = _1d(grid_size, d_w)

    pe = torch.zeros(n_temporal, grid_size, grid_size, dim)
    pe[..., :d_t] = pe_t[:, None, None, :]
    pe[..., d_t : d_t + d_h] = pe_h[None, :, None, :]
    pe[..., d_t + d_h :] = pe_w[None, None, :, :]
    return pe.reshape(n_temporal * grid_size * grid_size, dim)


class MultiBlockMask:
    """Samples a boolean mask over the (n_temporal, grid, grid) token grid:
    a handful of random rectangular spatial blocks, applied identically at
    every temporal tubelet, until roughly `mask_ratio` of tokens are
    masked. This is V-JEPA's block-masking idea simplified for an 8x8 grid.
    """

    def __init__(self, cfg: JepaConfig, n_blocks: int = 4):
        self.cfg = cfg
        self.n_blocks = n_blocks

    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        g = self.cfg.grid_size
        target = int(round(self.cfg.mask_ratio * g * g))
        masks = torch.zeros(batch_size, g, g, dtype=torch.bool, device=device)
        for b in range(batch_size):
            spatial = torch.zeros(g, g, dtype=torch.bool, device=device)
            attempts = 0
            while spatial.sum().item() < target and attempts < 50:
                attempts += 1
                bh = torch.randint(g // 4 + 1, g // 2 + 1, (1,)).item()
                bw = torch.randint(g // 4 + 1, g // 2 + 1, (1,)).item()
                top = torch.randint(0, g - bh + 1, (1,)).item()
                left = torch.randint(0, g - bw + 1, (1,)).item()
                spatial[top : top + bh, left : left + bw] = True
            # block sampling can overshoot/undershoot target by a few cells
            # (overlap, or attempts exhausted) -- every batch element needs
            # the exact same masked-token count so context-encoder outputs
            # stack into one tensor, so correct to `target` exactly.
            count = int(spatial.sum().item())
            if count > target:
                masked_idx = spatial.nonzero(as_tuple=False)
                drop = masked_idx[torch.randperm(count)[: count - target]]
                spatial[drop[:, 0], drop[:, 1]] = False
            elif count < target:
                unmasked_idx = (~spatial).nonzero(as_tuple=False)
                n_unmasked = unmasked_idx.shape[0]
                add = unmasked_idx[torch.randperm(n_unmasked)[: target - count]]
                spatial[add[:, 0], add[:, 1]] = True
            masks[b] = spatial
        # broadcast the same spatial mask across all temporal tubelets
        t = self.cfg.n_temporal_tokens
        masks = masks.unsqueeze(1).expand(batch_size, t, g, g)
        return masks.reshape(batch_size, t * g * g)


class Predictor(nn.Module):
    """Takes context-encoder tokens at visible positions + mask tokens (with
    positional embeddings) at masked positions, self-attends, and outputs a
    prediction for each masked position in encoder_dim space."""

    def __init__(self, cfg: JepaConfig):
        super().__init__()
        self.cfg = cfg
        self.in_proj = nn.Linear(cfg.encoder_dim, cfg.predictor_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.predictor_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.transformer = ViTEncoder(
            cfg.predictor_dim, cfg.predictor_depth, cfg.predictor_heads, cfg.mlp_ratio
        )
        self.out_proj = nn.Linear(cfg.predictor_dim, cfg.encoder_dim)
        pos = sincos_pos_embed(cfg.n_temporal_tokens, cfg.grid_size, cfg.predictor_dim)
        self.register_buffer("pos_embed", pos.unsqueeze(0))  # (1, n_tokens, dim)

    def forward(
        self, context_tokens: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """context_tokens: (B, n_visible, encoder_dim) tokens the encoder
        produced for visible positions only, already ordered by their
        original position among the visible set.
        mask: (B, n_tokens) bool, True = masked.
        Returns predictions for the masked positions: (B, n_masked, encoder_dim).
        """
        b, n_tokens = mask.shape
        d = self.cfg.predictor_dim
        full = torch.zeros(b, n_tokens, d, device=mask.device, dtype=context_tokens.dtype)
        ctx_proj = self.in_proj(context_tokens)
        for i in range(b):
            vis_idx = (~mask[i]).nonzero(as_tuple=True)[0]
            full[i, vis_idx] = ctx_proj[i, : vis_idx.numel()]
            masked_idx = mask[i].nonzero(as_tuple=True)[0]
            full[i, masked_idx] = self.mask_token.squeeze(0)
        full = full + self.pos_embed
        out = self.transformer(full)
        out = self.out_proj(out)
        preds = [out[i, mask[i]] for i in range(b)]
        return torch.stack(preds, dim=0)


class JEPA(nn.Module):
    def __init__(self, cfg: JepaConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = PatchEmbed3D(cfg, cfg.encoder_dim)
        self.context_encoder = ViTEncoder(
            cfg.encoder_dim, cfg.encoder_depth, cfg.encoder_heads, cfg.mlp_ratio
        )
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        pos = sincos_pos_embed(cfg.n_temporal_tokens, cfg.grid_size, cfg.encoder_dim)
        self.register_buffer("pos_embed", pos.unsqueeze(0))
        self.predictor = Predictor(cfg)
        self.masker = MultiBlockMask(cfg)

    @torch.no_grad()
    def update_target_encoder(self):
        m = self.cfg.ema_momentum
        for tp, cp in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            tp.mul_(m).add_(cp, alpha=1 - m)

    def encode_context(
        self, tokens: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """tokens: (B, n_tokens, dim) with pos_embed already added.
        Returns encoder output for visible tokens only, one tensor per batch
        row concatenated to the max visible count (assumes same mask ratio
        so counts match across the batch, true for MultiBlockMask)."""
        b = tokens.shape[0]
        n_visible = (~mask[0]).sum().item()
        visible = torch.zeros(
            b, n_visible, tokens.shape[-1], device=tokens.device, dtype=tokens.dtype
        )
        for i in range(b):
            vis_idx = (~mask[i]).nonzero(as_tuple=True)[0]
            visible[i] = tokens[i, vis_idx]
        return self.context_encoder(visible)

    def forward(self, clip: torch.Tensor):
        """clip: (B, T, C, H, W) in [0, 1]. Returns (loss, stats dict)."""
        b = clip.shape[0]
        raw_tokens = self.patch_embed(clip)  # (B, n_tokens, dim)
        tokens = raw_tokens + self.pos_embed

        mask = self.masker.sample(b, clip.device)  # (B, n_tokens) bool, True=masked

        with torch.no_grad():
            target_out = self.target_encoder(tokens)  # (B, n_tokens, dim)
            target_masked = torch.stack(
                [target_out[i, mask[i]] for i in range(b)], dim=0
            )

        context_out = self.encode_context(tokens, mask)  # (B, n_visible, dim)
        pred_masked = self.predictor(context_out, mask)  # (B, n_masked, dim)

        loss = F.smooth_l1_loss(pred_masked, target_masked.detach())
        stats = {
            "target_std": target_out.detach().std(dim=(0, 1)).mean().item(),
            "n_masked": mask[0].sum().item(),
            "n_visible": (~mask[0]).sum().item(),
        }
        return loss, stats
