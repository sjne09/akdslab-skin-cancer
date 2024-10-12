from typing import Tuple

import torch
from torch import nn


class MILAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
    ) -> None:
        """
        Attention-based MIL
        https://arxiv.org/pdf/1802.04712.

        Parameters
        ----------
        embed_dim : int
            The number of features in the input

        num_heads : int
            The number of attention heads
        """
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.Tanh(),
            nn.Linear(128, num_heads),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input data, shape (N, embed_dim)

        Returns
        -------
        torch.Tensor
            The output, shape (num_heads, embed_dim)

        torch.Tensor
            Attention weights, shape (num_heads, N)
        """
        # get attention scores
        att: torch.Tensor = self.attn(x)  # (N, num_heads)
        att = att.transpose(1, 0)  # (num_heads, N)
        att = att.softmax(1)

        # get output z by taking a weighted average of the instances in x
        # with weights = attention scores
        z = att @ x  # (num_heads, embed_dim)

        return z, att


class GatedMILAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int) -> None:
        """
        Attention-based MIL with gating mechanism
        https://arxiv.org/pdf/1802.04712

        Parameters
        ----------
        embed_dim : int
            The number of features in the input

        num_heads : int
            The number of attention heads
        """
        self.tanh = nn.Tanh()
        self.V = nn.Linear(embed_dim, 128)

        self.sigm = nn.Sigmoid()
        self.U = nn.Linear(embed_dim, 128)

        self.w = nn.Linear(128, num_heads)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input data, shape (N, embed_dim)

        Returns
        -------
        torch.Tensor
            The output, shape (num_heads, embed_dim)

        torch.Tensor
            Attention weights, shape (num_heads, N)
        """
        # get attention scores
        att_V: torch.Tensor = self.tanh(self.V(x))  # (N, 128)
        att_U: torch.Tensor = self.sigm(self.U(x))  # (N, 128)
        att = att_V * att_U
        att = self.w(att)  # (N, num_heads)
        att = att.transpose(1, 0)  # (num_heads, N)
        att = att.softmax(1)

        # get output z by taking a weighted average of the instances in x
        # with weights = attention scores
        z = att @ x  # (num_heads, embed_dim)

        return z, att
