import torch
from torch import nn

from .MLP import MLP


class AttentionAggregator(nn.Module):
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
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        # aggregator is a 1D "query" tensor, to be multiplied with the attn
        # output (in this case, attn produces the "key" tensor)
        self.aggregator = nn.Parameter(
            torch.nn.init.xavier_uniform_(torch.empty(1, embed_dim)).reshape(
                (embed_dim,)
            ),
            requires_grad=True,
        )
        self.mlp = MLP(
            in_features=embed_dim,
            hidden_dims=[embed_dim, embed_dim // 2, embed_dim // 4],
            out_features=out_features,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input data, shape (B, S, d_embed)

        Returns
        -------
        torch.Tensor
            The output, shape (B, d_out)
        """
        att, _ = self.attn(x, x, x)  # (B, S, d_embed)
        att = torch.matmul(self.aggregator, att.transpose(-2, -1))  # (B, S)

        # remove the sequence length dimension in favor of the embedding dim
        # input x is treated as the "value" tensor on attention outputs
        x = torch.matmul(att.unsqueeze(1), x).squeeze(1)  # (B, d_embed)

        x = self.mlp(x)
        return x
