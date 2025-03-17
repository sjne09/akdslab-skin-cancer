import os
import pickle
from typing import Dict

import numpy as np
import torch


def add_positions(
    slide_data: Dict[str, torch.Tensor], tile_size: int = 256
) -> None:
    """
    Adds relative positional data to the slide data based on coords. Operates
    in place.

    Parameters
    ----------
    slide_data : Dict[str, torch.Tensor]
        Tile embedding data for a single slide. Expected keys are
        "tile_embeds" and "coords"

    tile_size : int
        The size of the tile images used to generate the embeddings
    """
    # get the min x and y values. Can be negative due to padding added during
    # tiling procedure
    min_x, min_y = slide_data["coords"].min(dim=0)[0]

    # treating the min values as origin, adjust each coord by the respective
    # min value and divide by the tile size to get relative positions
    pos_x = (slide_data["coords"][:, 0] - min_x) / tile_size
    pos_y = (slide_data["coords"][:, 1] - min_y) / tile_size

    slide_data["pos"] = torch.stack((pos_x, pos_y), dim=1).to(torch.long)


def sort_embeds(slide_data: Dict[str, torch.Tensor]) -> None:
    """
    Sort tile embeddings based on coordinates (left to right, bottom to top).
    Operates in place.

    Parameters
    ----------
    slide_data : Dict[str, torch.Tensor]
        Tile embedding data for a single slide. Expected keys are
        "tile_embeds" and "coords"
    """
    # sort first on y, then on x
    # use lexsort for stability
    sort_order = np.lexsort(
        [slide_data["coords"][:, 1], slide_data["coords"][:, 0]]
    )
    slide_data["coords"] = slide_data["coords"][sort_order]
    slide_data["tile_embeds"] = slide_data["tile_embeds"][sort_order]


def postproc_pkl(
    input_dir: str, output_dir: str, tile_size: int = 256
) -> None:
    """
    Sorts tile embeddings by coordinates and adds relative positional data,
    then re-pickles the data.

    Parameters
    ----------
    input_dir : str
        The directory containing the saved tile embeddings. Directory should
        contain one .pkl file for each slide, and the pickle file should
        contain a dict with keys "tile_embeds" and "coords", for which the
        values are tensors with the applicable data

    output_dir : str
        Location to save the modified pickle files to

    tile_size : int
        The size of the tile images used to generate the embeddings
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # load each pickle file in the input_dir
    for fname in os.listdir(input_dir):
        if fname.endswith(".pkl"):
            fpath = os.path.join(input_dir, fname)
            with open(fpath, "rb") as f:
                embeds = pickle.load(f)

            # sort, then add positional information
            sort_embeds(embeds)
            add_positions(embeds, tile_size)

            # save to output dir
            with open(os.path.join(output_dir, fname), "wb") as f:
                pickle.dump(embeds, f)


def postproc_dict(
    embeds: Dict[str, torch.Tensor], tile_size: int = 256
) -> Dict[str, torch.Tensor]:
    """
    Sorts tile embeddings by coordinates and adds relative positional data.

    Parameters
    ----------
    embeds : Dict[str, torch.Tensor]
        Tile embedding data for a single slide. Expected keys are
        "tile_embeds" and "coords"

    tile_size : int
        The size of the tile images used to generate the embeddings

    Returns
    -------
    Dict[str, torch.Tensor]
        The updated tile embedding data with keys "tile_embeds", "coords",
        and "pos"
    """
    sort_embeds(embeds)
    add_positions(embeds, tile_size)
    return embeds
