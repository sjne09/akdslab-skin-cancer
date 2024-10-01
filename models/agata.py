import math

import torch
from torch import nn
from torch.nn import functional as F

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
        in_features: int,
        dim_k: int = 256,
        dim_v: int = 512,
        scaling: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        in_features : int
            The number of features in the input

        dim_k : int
            The dimension of the key parameter

        v_features : int
            The dimension of the value parameter

        scaling : bool
            Whether to scale the attention scores by 1/sqrt(dim_k)
        """
        super().__init__()
        self.gelu = nn.GELU()
        self.scaler = math.sqrt(dim_k) if scaling else 1.0

        self.w_k = nn.Linear(
            in_features=in_features, out_features=dim_k, bias=True
        )
        self.w_v = nn.Linear(in_features=dim_k, out_features=dim_v, bias=True)
        w_q = torch.empty((dim_k))
        self.q = nn.Parameter(torch.nn.init.uniform_(w_q), requires_grad=True)

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

        att = torch.matmul(self.q, k.transpose(-2, -1))  # (B, 1, S)
        att *= 1.0 / self.scaler
        att = F.softmax(att, dim=-1)

        y = torch.matmul(att, v)  # (B, 1, d_v)
        return y


class MLP(nn.Module):
    """
    Basic MLP with one hidden layer using the GELU nonlinearity.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        """
        Parameters
        ----------
        in_features : int
            The number of features in the input

        out_features : int
            The number of features in the output
        """
        super().__init__()
        self.fc_h = nn.Linear(in_features, in_features)
        self.gelu = nn.GELU()
        self.fc_o = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input data, shape (B, S, d_in)

        Returns
        -------
        torch.Tensor
            The output, shape (B, S, d_out)
        """
        x = self.gelu(self.fc_h(x))
        x = self.gelu(self.fc_o(x))
        return x


class AgataAggregator(nn.Module):
    """
    Modified from the architecture shown in Virchow to have only a single
    "Agata Head" with an output feature for each label, rather than having
    one head per label.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dim_k: int,
        dim_v: int,
        scaling: bool,
    ) -> None:
        """
        Parameters
        ----------
        in_features : int
            The number of features in the input

        out_features : int
            The number of features in the output

        dim_k : int
            The dimension of the key parameter

        v_features : int
            The dimension of the value parameter

        scaling : bool
            Whether to scale the attention scores by 1/sqrt(dim_k)
        """
        super().__init__()
        self.attn = AggregatorAttention(
            in_features, dim_k=dim_k, dim_v=dim_v, scaling=scaling
        )
        self.mlp = MLP(in_features=dim_v, out_features=out_features)

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
            The output of the attention layer, shape (B, 1, d_out)
        """
        x = self.attn(x)
        x = self.mlp(x)
        x = F.softmax(x)
        return x
