"""Small decoder mapping a single spatial grid of token embeddings back to
an approximate RGB frame. Purely a visualization/debugging tool -- lets us
*see* what a representation (real or predicted) corresponds to, the way
Dreamer's imagined-rollout figures or a VAE's reconstructions do. Not part
of the actual JEPA training objective (which deliberately predicts in
representation space, never pixels) and not needed for anything else in
the pipeline.
"""
import torch
import torch.nn as nn


class FrameDecoder(nn.Module):
    """(B, grid_size*grid_size, embed_dim) -> (B, 3, img_size, img_size).
    One spatial tubelet's worth of tokens (e.g. the last temporal
    position), not a full clip -- decoding a single frame is enough to see
    what a representation encodes/predicts.
    """

    def __init__(self, embed_dim: int = 768, grid_size: int = 8, img_size: int = 128):
        super().__init__()
        self.grid_size = grid_size
        n_upsamples = 0
        size = grid_size
        while size < img_size:
            size *= 2
            n_upsamples += 1
        chans = [embed_dim] + [max(32, embed_dim // (2 ** (i + 1))) for i in range(n_upsamples)]
        layers = []
        for i in range(n_upsamples):
            layers += [
                nn.ConvTranspose2d(chans[i], chans[i + 1], kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(8, chans[i + 1]),
                nn.GELU(),
            ]
        layers += [nn.Conv2d(chans[-1], 3, kernel_size=3, padding=1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, grid_size*grid_size, D) -> (B, 3, img_size, img_size) in [0, 1]."""
        b, n, d = tokens.shape
        assert n == self.grid_size * self.grid_size, f"expected {self.grid_size**2} tokens, got {n}"
        x = tokens.transpose(1, 2).reshape(b, d, self.grid_size, self.grid_size)
        return self.net(x)


def last_tubelet_tokens(tokens: torch.Tensor, grid_size: int) -> torch.Tensor:
    """tokens: (B, T*grid_size*grid_size, D), temporal-major flatten (matches
    PatchEmbed3D/the pretrained ViT's own patchify order) -> last temporal
    tubelet's spatial grid, (B, grid_size*grid_size, D)."""
    return tokens[:, -grid_size * grid_size:, :]
