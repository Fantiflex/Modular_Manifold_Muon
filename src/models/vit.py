"""
Small Vision Transformer used in the Manifold MuOn experiments.
"""

import torch
import torch.nn as nn


class SimpleViT(nn.Module):
    """
    Small Vision Transformer for CIFAR-sized images.

    Default configuration:
        image size: 32x32
        patch size: 4x4
        number of patches: 64
        embedding dimension: 128
        transformer depth: 2
        attention heads: 4
        MLP ratio: 4
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_chans: int = 3,
        num_classes: int = 100,
        embed_dim: int = 128,
        depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans

        num_patches = (img_size // patch_size) ** 2
        patch_dim = in_chans * patch_size * patch_size

        self.patch_embed = nn.Linear(
            patch_dim,
            embed_dim,
            bias=False,
        )

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
        )

        self.norm = nn.LayerNorm(embed_dim)

        self.head = nn.Linear(
            embed_dim,
            num_classes,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        assert H == self.img_size and W == self.img_size

        p = self.patch_size

        x = x.reshape(
            B,
            C,
            H // p,
            p,
            W // p,
            p,
        )

        x = x.permute(
            0,
            2,
            4,
            1,
            3,
            5,
        ).contiguous()

        x = x.view(
            B,
            -1,
            C * p * p,
        )

        x = self.patch_embed(x)

        x = x + self.pos_embed

        x = self.encoder(x)

        x = x.mean(dim=1)

        x = self.norm(x)

        return self.head(x)