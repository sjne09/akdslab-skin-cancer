from typing import List

import torch
from torch import nn


class MLP(nn.Module):
    """
    Basic MLP with one hidden layer using the GELU nonlinearity.
    """

    def __init__(
        self, in_features: int, hidden_dims: List[int], out_features: int
    ) -> None:
        """
        Parameters
        ----------
        in_features : int
            The number of features in the input

        hidden_dims : List[int]
            A list of dimensions to be used for hidden layers

        out_features : int
            The number of features in the output
        """
        super().__init__()
        self.gelu = nn.GELU()

        hidden = []
        prev_dim = in_features
        for dim in hidden_dims:
            hidden.append(nn.Linear(prev_dim, dim))
            prev_dim = dim
        self.h = nn.ModuleList(hidden)

        self.fc_o = nn.Linear(in_features // 4, out_features)

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
        for lin in self.h:
            x = self.gelu(lin(x))
        x = self.fc_o(x)
        return x
