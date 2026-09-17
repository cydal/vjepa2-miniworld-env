"""Action-conditioned predictor for Phase 3 (post-training), matching the
real V-JEPA2-AC design (src/models/ac_predictor.py in facebookresearch/vjepa2):
two extra tokens per temporal tubelet -- action (the control command) and
state (the agent's own proprioceptive configuration) -- with block-causal
attention (a tubelet's tokens may attend to itself and all earlier
tubelets, never later ones), predicting only the visual tokens of each
tubelet from everything before it.

For this car-nav domain:
  action (3-dim): [throttle, brake, steer] -- the control command applied.
  state  (3-dim): [speed, steer_angle, heading] -- the car's own physical
    configuration. Deliberately excludes absolute (x, y): each episode's
    city is an independently randomized layout, so raw world position has
    no cross-episode correspondence (unlike a robot's fixed base frame,
    where end-effector pose IS consistently meaningful across episodes).
    Also excludes yaw_rate: it's a deterministic function of speed and
    steer_angle already in state (see env/car.py), so it adds no
    information -- not recorded in meta.json, and there'd be nothing to
    gain by adding it.
  target/goal information is deliberately NOT part of either -- it belongs
  at planning time (CEM against a goal embedding), never baked into
  training. See notes/journal.md and docs/phase2-plan.md for the reasoning.

This module is intentionally separate from model/jepa.py rather than
extending TransformerBlock in place, so it doesn't touch anything the
current (running) encoder fine-tune depends on.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from model.jepa import JepaConfig, sincos_pos_embed

ACTION_DIM = 3   # [throttle, brake, steer]
STATE_DIM = 3    # [speed, steer_angle, heading]
COND_TOKENS = 2  # action + state, per tubelet


def build_block_causal_mask(n_temporal: int, grid_size: int, cond_tokens: int = COND_TOKENS) -> torch.Tensor:
    """(L, L) bool mask, True = *not* attendable (matches torch's
    attn_mask convention for MultiheadAttention: True positions are
    masked out). Tokens are grouped per tubelet as
    [action, state, H*W visual tokens], length (cond_tokens + grid^2) each,
    n_temporal tubelets total. A tubelet's tokens may attend to their own
    tubelet and all earlier ones, never later ones -- block-causal, not
    fully causal within a tubelet (order within a tubelet doesn't encode
    any real temporal information, unlike across tubelets)."""
    per_tubelet = cond_tokens + grid_size * grid_size
    L = n_temporal * per_tubelet
    tubelet_of = torch.arange(L) // per_tubelet
    # mask[i, j] = True (blocked) if j's tubelet is AFTER i's tubelet
    mask = tubelet_of.unsqueeze(1) < tubelet_of.unsqueeze(0)
    return mask


class CausalBlock(nn.Module):
    """Same shape as model.jepa.TransformerBlock, with attn_mask support
    (the one thing the plain-pretraining block doesn't need)."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class ACPredictor(nn.Module):
    """context tokens (from the frozen, fine-tuned encoder, FULL clip --
    no masking on the encoder side for AC training, unlike Phase 2's
    masked-prediction task) + per-tubelet action/state -> predicted visual
    tokens for every tubelet, each conditioned only on itself and the past.
    """

    def __init__(self, cfg: JepaConfig, predictor_dim: int = 384, depth: int = 6,
                 heads: int = 6, mlp_ratio: float = 4.0,
                 action_dim: int = ACTION_DIM, state_dim: int = STATE_DIM):
        super().__init__()
        self.cfg = cfg
        self.predictor_dim = predictor_dim
        self.in_proj = nn.Linear(cfg.encoder_dim, predictor_dim)
        self.action_encoder = nn.Linear(action_dim, predictor_dim)
        self.state_encoder = nn.Linear(state_dim, predictor_dim)
        self.blocks = nn.ModuleList(
            [CausalBlock(predictor_dim, heads, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(predictor_dim)
        self.out_proj = nn.Linear(predictor_dim, cfg.encoder_dim)

        pos = sincos_pos_embed(cfg.n_temporal_tokens, cfg.grid_size, predictor_dim)
        pos = pos.view(cfg.n_temporal_tokens, cfg.grid_size * cfg.grid_size, predictor_dim)
        self.register_buffer("visual_pos_embed", pos)  # (T, H*W, D), added to visual tokens only

        mask = build_block_causal_mask(cfg.n_temporal_tokens, cfg.grid_size)
        self.register_buffer("attn_mask", mask)

    def forward(self, visual_tokens: torch.Tensor, actions: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """visual_tokens: (B, T*H*W, encoder_dim) -- the FULL context clip's
        encoder output, T=n_temporal_tokens tubelets of H*W tokens each.
        actions: (B, T, action_dim), states: (B, T, state_dim) -- one
        action/state pair per tubelet (see model/ac_dataset.py for how these
        get aligned to tubelets from raw per-step records).
        Returns predicted encoder-space representations, (B, T*H*W, encoder_dim).
        """
        cfg = self.cfg
        B, N, _ = visual_tokens.shape
        T, HW = cfg.n_temporal_tokens, cfg.grid_size * cfg.grid_size
        assert N == T * HW

        v = self.in_proj(visual_tokens).view(B, T, HW, self.predictor_dim)
        v = v + self.visual_pos_embed.unsqueeze(0)
        a = self.action_encoder(actions).unsqueeze(2)  # (B, T, 1, D)
        s = self.state_encoder(states).unsqueeze(2)     # (B, T, 1, D)

        x = torch.cat([a, s, v], dim=2)  # (B, T, 2+HW, D)
        x = x.flatten(1, 2)              # (B, T*(2+HW), D)

        for blk in self.blocks:
            x = blk(x, attn_mask=self.attn_mask)
        x = self.norm(x)

        x = x.view(B, T, 2 + HW, self.predictor_dim)
        visual_out = x[:, :, 2:, :].flatten(1, 2)  # drop action/state positions
        return self.out_proj(visual_out)
