import torch
from torch import nn

from .abmil import GatedMILAttention, MILAttention
from .MLP import MLP
from .pos_embedding import PositionalEmbedding


class MILClassifier(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        out_features: int,
        num_heads: int,
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
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.pos = PositionalEmbedding(embed_dim)

        if gated:
            self.attn = GatedMILAttention(embed_dim, num_heads)
        else:
            self.attn = MILAttention(embed_dim, num_heads)
        self.proj = nn.Linear(embed_dim * num_heads, embed_dim)
        self.mlp = MLP(
            in_features=embed_dim,
            hidden_dims=[1024, 512, 256],
            out_features=out_features,
        )

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
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
        B, S, D = x.shape
        x = x + self.pos(coords)  # (B, S, d_embed)
        x, att_weights = self.attn(x)  # (B, num_heads, d_embed)
        x = self.proj(
            x.view(B, self.embed_dim * self.num_heads)
        )  # (B, d_embed)
        x = self.mlp(x)  # (B, d_out)
        return x
