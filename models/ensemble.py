from typing import Dict

import torch
from torch import nn

from .abmil import GatedMILAttention, MILAttention
from .MLP import MLP
from .pos_embedding import PositionalEmbedding


class EnsembleClassifier(nn.Module):
    def __init__(
        self,
        uni_embed_dim: int = 1024,
        gigapath_embed_dim: int = 1536,
        prism_embed_dim: int = 1280,
        out_features: int = 4,
        num_heads: int = 1,
        gated: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        embed_dim : int
            The number of features in the input

        out_features : int
            The number of features in the output

        num_heads : int
            The number of attention heads

        gated : bool
            Whether to use the gated MIL architecture
        """
        super().__init__()
        self.uni_embed_dim = uni_embed_dim
        self.gigapath_embed_dim = gigapath_embed_dim
        self.prism_embed_dim = prism_embed_dim
        self.total_embed_dim = (
            uni_embed_dim + gigapath_embed_dim + prism_embed_dim
        )
        self.num_heads = num_heads

        self.uni_pos = PositionalEmbedding(uni_embed_dim)
        self.gigapath_pos = PositionalEmbedding(gigapath_embed_dim)

        if gated:
            self.uni_attn = GatedMILAttention(uni_embed_dim, num_heads)
            self.gigapath_attn = GatedMILAttention(
                gigapath_embed_dim, num_heads
            )
        else:
            self.uni_attn = MILAttention(uni_embed_dim, num_heads)
            self.gigapath_attn = MILAttention(gigapath_embed_dim, num_heads)

        self.uni_proj = nn.Linear(uni_embed_dim * num_heads, uni_embed_dim)
        self.gigapath_proj = nn.Linear(
            gigapath_embed_dim * num_heads, gigapath_embed_dim
        )
        self.mlp = MLP(
            in_features=self.total_embed_dim,
            hidden_dims=[1024, 512, 256],
            out_features=out_features,
        )

    def forward(
        self, x: Dict[str, torch.Tensor], coords: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Performs a forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input data, shape (B, S, d_embed)

        coords : torch.Tensor
            Coordinates matching to input tensor, shape (B, S, 2)

        Returns
        -------
        torch.Tensor
            The output, shape (B, d_out)
        """
        B = x["uni"].shape[0]
        x_uni = x["uni"] + self.uni_pos(coords["uni"])  # (B, S, d_embed)
        x_gigapath = x["gigapath"] + self.gigapath_pos(
            coords["gigapath"]
        )  # (B, S, d_embed)

        x_uni, _ = self.uni_attn(x_uni)  # (B, num_heads, d_embed)
        x_gigapath, _ = self.gigapath_attn(x_gigapath)

        x_uni = self.uni_proj(
            x_uni.view(B, self.uni_embed_dim * self.num_heads)
        )
        x_gigapath = self.gigapath_proj(
            x_gigapath.view(B, self.gigapath_embed_dim * self.num_heads)
        )

        x = torch.cat([x_uni, x_gigapath, x["prism"]], dim=-1)  # (B, d_total)
        x = self.mlp(x)  # (B, d_out)
        return x
