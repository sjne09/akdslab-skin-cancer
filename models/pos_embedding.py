from math import log
from typing import Union

import numpy as np
import torch
from torch import nn


class PositionalEmbedding(nn.Module):
    """
    References:
    Pytorch
    prov-gigapath
    """

    def __init__(self, embed_dim: int, max_len: int = 10000) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        dim = embed_dim // 2
        div_term = torch.exp(torch.arange(0, dim, 2) * (-log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, coords):
        x_ranks = get_position_ranks(coords[:, :, 0])
        y_ranks = get_position_ranks(coords[:, :, 1])
        x_embed = self.pe[x_ranks]
        y_embed = self.pe[y_ranks]
        return torch.cat((x_embed, y_embed), dim=-1)


def get_position_ranks(
    x: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """ """
    if isinstance(x, np.ndarray):
        return _get_position_ranks_np(x)
    elif isinstance(x, torch.Tensor):
        return _get_position_ranks_torch(x)
    else:
        raise ValueError("incorrect type")


def _get_position_ranks_np(x: np.ndarray) -> np.ndarray:
    """Taken from scipy.stats.rankdata"""
    j = np.argsort(x, axis=-1)
    y = np.take_along_axis(x, j, axis=-1)
    i = np.concatenate(
        [
            np.ones(x.shape[:-1] + (1,), dtype=np.bool_),
            y[..., :-1] != y[..., 1:],
        ],
        axis=-1,
    )
    # get integer indices of unique elements
    indices = np.arange(y.size)[i.ravel()]
    # get counts of unique elements
    counts = np.diff(indices, append=y.size)
    ranks = np.cumsum(i, axis=-1)[i]

    ranks = np.repeat(ranks, counts).reshape(x.shape)
    ordered_ranks = np.empty(j.shape, dtype=ranks.dtype)
    np.put_along_axis(ordered_ranks, j, ranks, axis=-1)
    return ordered_ranks - 1


def _get_position_ranks_torch(x: torch.Tensor) -> torch.Tensor:
    """Adapted from scipy.stats.rankdata"""
    # get the sorting indices and sort x
    j = torch.argsort(x, dim=-1)
    y = torch.gather(x, -1, j)

    # create boolean mask for unique elements
    i = torch.cat(
        [
            torch.ones(x.shape[:-1] + (1,), dtype=torch.bool, device=x.device),
            y[..., :-1] != y[..., 1:],
        ],
        dim=-1,
    )

    # get integer indices of unique elements
    indices = torch.arange(y.numel(), device=x.device)[i.view(-1)]

    # get counts of unique elements
    counts = torch.diff(
        indices, append=torch.tensor([y.numel()], device=x.device)
    )
    ranks = torch.cumsum(i, dim=-1)[i]

    # expand the ranks to the full size of the input
    ranks = ranks.repeat_interleave(counts).view(x.shape)
    ordered_ranks = torch.empty_like(j, dtype=ranks.dtype, device=x.device)

    # place ranks in the correct positions
    ordered_ranks.scatter_(-1, j, ranks)

    return ordered_ranks - 1


# enc = PositionalEmbedding(1024, 9000)
# x = torch.rand((2, 2500, 1024))
# coords = torch.randint(0, 50, (2, 2500, 2))
# pos = enc(coords)
# print(pos.shape)
# print(x + pos)

# import matplotlib.pyplot as plt

# fig = plt.figure(figsize=(10, 8))
# plt.imshow(x[8001:9000, :], cmap="PuOr_r")
# plt.savefig("pos.png")
