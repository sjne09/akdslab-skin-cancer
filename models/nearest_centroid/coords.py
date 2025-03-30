import os
from typing import Dict, List, Tuple

from data_processing.slide_utils import load_slide


def map_coords(
    slide_path: str, tiles_path: str
) -> Dict[Tuple[int, int], Tuple[int, int]]:
    """
    Maps tile coords from the original WSI to coords for the tiles
    derived from a processed slide

    Parameters
    ----------
    slide_path : str
        Path to the original WSI

    tiles_path : str
        Path to the directory containing tiles corresponding to the WSI

    Returns
    -------
    Dict[Tuple[int, int], Tuple[int, int]]
        A mapping of original coordinates to coordinates of tiles from
        the processed WSI
    """
    img = load_slide(slide_path)
    coords = get_tile_coords(tiles_path)
    # map the original coords to the modified coords
    coord_mapping = {
        calculate_original_coords(pair, img["origin"], img["scale"]): pair
        for pair in coords
    }
    return coord_mapping


def calculate_original_coords(
    coords: Tuple[int, int],
    origin: Tuple[int, int],
    downscale_factor: float,
) -> Tuple[int, int]:
    """
    Converts coords from a processed WSI to coords on the original WSI.

    Parameters
    ----------
    coords : Tuple[int, int]
        Coords to convert

    origin : Tuple[int, int]
        The coordinates of the origin for the processed slide

    downscale_factor : float
        The downscale factor for the processed slide

    Returns
    -------
    Tuple[int, int]
        The coordinates to convert
    """
    return (
        int((coords[0] - origin[0]) / downscale_factor),
        int((coords[1] - origin[1]) / downscale_factor),
    )


def get_tile_coords(tiles_path: str) -> List[Tuple[int, int]]:
    """
    Gets coordinates from file names contained in tiles_path.

    Parameters
    ----------
    tiles_path : str
        Path to a directory containing slide tiles

    Returns
    -------
    List[Tuple[int, int]]
        Coordinates for all tiles in tiles_path in (x, y) pairs
    """
    tiles = [
        tile[:-4] for tile in os.listdir(tiles_path) if tile.endswith(".png")
    ]

    coords = []
    for tile in tiles:
        x, y = tile.split("_")
        x, y = int(x.replace("x", "")), int(y.replace("y", ""))
        coords.append((x, y))
    return coords
