import torch
from torch import nn

from .abmil import MILAttention
from .MLP import MLP
from .pos_embedding import PositionalEmbedding


class MILClassifier(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        out_features: int,
        num_heads: int,
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
        """
        super().__init__()
        self.pos = PositionalEmbedding(embed_dim)

        # could add linear layer prior to attention
        self.attn = MILAttention(embed_dim, num_heads)
        self.mlp = MLP(
            in_features=embed_dim,
            hidden_dims=[512, 256, 128],
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
        x = x + self.pos(coords)  # (B, S, d_embed)
        x, att_weights = self.attn(x)  # (B, num_heads, d_embed)
        x = self.mlp(x)
        return x
