from math import log
from typing import Union

import numpy as np
import torch
from torch import nn


class PositionalEmbedding(nn.Module):
    """
    2D sinusoidal positional embeddings.

    References
    Pytorch:
        https://pytorch.org/tutorials/beginner/translation_transformer.html
    Prov-GigaPath: https://github.com/prov-gigapath/prov-gigapath/
    """

    def __init__(self, embed_dim: int, max_len: int = 500) -> None:
        super().__init__()
        position = torch.arange(max_len)

        dim = embed_dim // 2
        div_term = torch.exp(torch.arange(0, dim, 2) * (-log(10000.0) / dim))

        # for each position, multiply that position by the div_term vector
        out = torch.outer(position, div_term)
        pe_sin = torch.sin(out)
        pe_cos = torch.cos(out)
        pe = torch.cat((pe_sin, pe_cos), axis=1)
        self.register_buffer("pe", pe)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        coords : torch.Tensor
            Tile cartesian coordinates, shape (B, S, 2) or (N, 2)

        Returns
        -------
        torch.Tensor
            Positional embeddings for the input coords
        """
        x_embed = self.pe[coords[..., 0]]
        y_embed = self.pe[coords[..., 1]]
        return torch.cat((y_embed, x_embed), dim=-1)


def get_position_ranks(
    x: Union[np.ndarray, torch.Tensor],
) -> Union[np.ndarray, torch.Tensor]:
    """
    Get position ranks from coordinates.

    Parameters
    ----------
    x : Union[np.ndarray, torch.Tensor]
        A 2D iterable containing cartesian coordinate pairs

    Returns
    -------
    Union[np.ndarray, torch.Tensor]
        Position rank pairs corresponding to input coordinate pairs
    """
    if isinstance(x, np.ndarray):
        return _get_position_ranks_np(x)
    elif isinstance(x, torch.Tensor):
        return _get_position_ranks_torch(x)
    else:
        raise ValueError("incorrect type")


def _get_position_ranks_np(x: np.ndarray) -> np.ndarray:
    """
    Taken from scipy.stats.rankdata
    Get position ranks from coordinates.

    Parameters
    ----------
    x : np.ndarray
        A 2D array containing cartesian coordinate pairs

    Returns
    -------
    np.ndarray
        Position rank pairs corresponding to input coordinate pairs
    """
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

    # expand the ranks to the full size of the input
    ranks = np.repeat(ranks, counts).reshape(x.shape)
    ordered_ranks = np.empty(j.shape, dtype=ranks.dtype)

    # place ranks in the correct positions
    np.put_along_axis(ordered_ranks, j, ranks, axis=-1)

    # zero indexing
    return ordered_ranks - 1


def _get_position_ranks_torch(x: torch.Tensor) -> torch.Tensor:
    """
    Adapted from scipy.stats.rankdata
        Get position ranks from coordinates.

    Parameters
    ----------
    x : torch.Tensor
        A 2D tensor containing cartesian coordinate pairs

    Returns
    -------
    torch.Tensor
        Position rank pairs corresponding to input coordinate pairs
    """
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

    # zero indexing
    return ordered_ranks - 1
