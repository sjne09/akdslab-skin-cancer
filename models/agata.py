import math

import torch
from torch import nn
from torch.nn import functional as F

from .MLP import MLP

"""
Implements the Agata Aggregator model as used in Virchow for generating
specimen-level predictions (https://arxiv.org/pdf/2309.07778).
"""


class AggregatorAttention(nn.Module):
    """
    Cross attention layer. Uses a fixed query parameter to reduce memory
    consumption. Additionally calculates keys directly from the input, but
    feeds through a GELU nonlinearity. It then calculates values from the
    keys (rather than directly from the input) and applies a nonlinearity
    to the value tensor as well.
    """

    def __init__(
        self,
        embed_dim: int,
        kdim: int = 256,
        vdim: int = 512,
        scaling: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        embed_dim : int
            The number of features in the input

        kdim : int
            The dimension of the key parameter

        vdim : int
            The dimension of the value parameter

        scaling : bool
            Whether to scale the attention scores by 1/sqrt(dim_k)
        """
        super().__init__()
        self.gelu = nn.GELU()
        self.scaler = math.sqrt(kdim) if scaling else 1.0

        self.w_k = nn.Linear(
            in_features=embed_dim, out_features=kdim, bias=True
        )
        self.w_v = nn.Linear(in_features=kdim, out_features=vdim, bias=True)
        self.q = nn.Parameter(
            torch.nn.init.xavier_uniform_(torch.empty(1, kdim)).reshape(
                (kdim,)
            ),
            requires_grad=True,
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
            The output of the attention layer, shape (B, 1, d_v)
        """
        k = self.gelu(self.w_k(x))  # (B, S, d_k)
        v = self.gelu(self.w_v(k))  # (B, S, d_v)

        att = torch.matmul(self.q, k.transpose(-2, -1))  # (B, S)
        att *= 1.0 / self.scaler
        att = F.softmax(att, dim=-1)

        y = torch.matmul(att.unsqueeze(1), v).squeeze(1)  # (B, d_v)
        return y


class AgataAggregator(nn.Module):
    """
    Modified from the architecture shown in Virchow to have only a single
    "Agata Head" with an output feature for each label, rather than having
    one head per label.
    """

    def __init__(
        self,
        embed_dim: int,
        out_features: int,
        kdim: int,
        vdim: int,
        scaling: bool,
    ) -> None:
        """
        Parameters
        ----------
        embed_dim : int
            The number of features in the input

        out_features : int
            The number of features in the output

        kdim : int
            The dimension of the key parameter

        vdim : int
            The dimension of the value parameter

        scaling : bool
            Whether to scale the attention scores by 1/sqrt(dim_k)
        """
        super().__init__()
        self.attn = AggregatorAttention(
            embed_dim, kdim=kdim, vdim=vdim, scaling=scaling
        )
        self.mlp = MLP(
            in_features=vdim,
            hidden_dims=[vdim, vdim // 2, vdim // 4],
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
        x = self.attn(x)
        x = self.mlp(x)
        return x
