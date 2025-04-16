import os
from operator import itemgetter
from typing import Callable, Dict, List, Set, Tuple, Union

import geojson
import torch
from shapely import Geometry, MultiPolygon, Polygon, box


def extract_relevant_tiles(
    polygon: Polygon,
    coords: List[Tuple[int, int]],
    relation_func: Callable[[Geometry, Geometry], bool],
    tile_size: int = 256,
) -> Set[Tuple[int, int]]:
    """
    Determines which tiles from the list of tile coordinates have the
    spatial relationship specified by relation_func with the input
    polygon and returns the list of relevant tiles as a set.

    Parameters
    ----------
    polygon : shapely.Polygon
        The polygon to use as a bounding box

    coords : List[Tuple[int, int]]
        A list containing (x, y) coordinate tuples

    relation_func : Callable[[Geometry, Geometry], bool]
        A function that tests a spatial relationship between two shapes.
        When True, the tile being tested will be added to the output
        set

    tile_size : int
        The tile size

    Returns
    -------
    Set[Tuple[int, int]]
        Relevant tiles that intersect the polygon
    """
    relevant_tiles = []
    for xmin, ymin in coords:
        xmax, ymax = xmin + tile_size, ymin + tile_size
        tile = box(xmin, ymin, xmax, ymax)
        if relation_func(polygon, tile):
            relevant_tiles.append((xmin, ymin))

    return set(relevant_tiles)


def extract_polygon(coords_list: List[List[float]]) -> Polygon:
    """
    Extract a Polygon object from a list of coordinates.

    Parameters
    ----------
    coords_list : List[List[float]]
        The list of coordinates for a polygon in geojson format

    Returns
    -------
    shapely.Polygon
        The polygon corresponding to the provided coordinates
    """
    exterior = None
    interior = []
    for i, coords in enumerate(coords_list):
        if i == 0:
            exterior = coords
        else:
            interior.append(coords)
    return Polygon(exterior, interior)


def extract_shape(shape: geojson.Feature) -> Union[MultiPolygon, Polygon]:
    """
    Extract a shapely shape from a geojson feature object.

    Parameters
    ----------
    shape : geojson.Feature
        The feature object containing the shape specifications

    Returns
    -------
    Union[MultiPolygon, Polygon]
        The shapely shape corresponding to the input feature
    """
    shape_type = shape["geometry"]["type"]
    coords = shape["geometry"]["coordinates"]

    if shape_type == "Polygon":
        polygon = extract_polygon(coords)
    elif shape_type == "MultiPolygon":
        mp = []
        for sub_polygon in coords:
            mp.append(extract_polygon(sub_polygon))
        polygon = MultiPolygon(mp)

    return polygon


def get_subclass_polygons(
    root: str,
) -> Dict[str, Dict[str, List[Polygon]]]:
    """
    Get polygons for slides containing a subclass in the shape's
    ["properties"]["name"] object. Returns a dict with keys for each
    subclass and dict values with slide ids as keys and a list of
    polygons as values.

    Parameters
    ----------
    root : str
        The directory containing geojson files for a particular class.
        Each geojson should contain polygon data for a single slide

    Returns
    -------
    Dict[str, Dict[str, List[Polygon]]]
        A dict with keys for each subclass and dict values with slide ids
        as keys and a list of polygons as values
    """
    # for each slide in the sampled specimen
    polygons = {}
    for slide in os.listdir(root):
        slide_name = os.path.splitext(slide)[0]
        fpath = os.path.join(root, slide)
        features = get_geojson_features(fpath)

        # for each shape in the geojson, extract the tissue type and add
        # the polygon to the corresponding entry in geoms
        for shape in features:
            tissue_type = shape["properties"]["name"]
            polygon = extract_shape(shape)

            if tissue_type not in polygons:
                polygons[tissue_type] = {slide_name: [polygon]}
            elif slide_name not in polygons[tissue_type]:
                polygons[tissue_type][slide_name] = [polygon]
            else:
                polygons[tissue_type][slide_name].append(polygon)
    return polygons


def get_class_polygons(root: str) -> Dict[str, List[Polygon]]:
    """
    Get polygons for slides of a certain class. Returns a dict with slide
    ids as keys and a list of polygons as values.

    Parameters
    ----------
    root : str
        The directory containing geojson files for a particular class.
        Each geojson should contain polygon data for a single slide

    Returns
    -------
    Dict[str, List[Polygon]]
        A dict with slide ids as keys and a list of polygons as values
    """
    polygons = {}
    for slide in os.listdir(root):
        if not slide.endswith(".geojson"):
            continue
        slide_name = os.path.splitext(slide)[0]
        fpath = os.path.join(root, slide)
        features = get_geojson_features(fpath)

        # for each shape in the geojson, extract the shape as a shapely
        # object and append to list of geometries
        geoms = []
        for shape in features:
            if shape["geometry"]["type"] not in {
                "Polygon",
                "MultiPolygon",
            }:
                continue
            polygon = extract_shape(shape)
            geoms.append(polygon)
        polygons[slide_name] = geoms

    return polygons


def get_geojson_features(fpath: str) -> List[geojson.feature.Feature]:
    """
    Get the features contained in a geojson file.

    Parameters
    ----------
    fpath : str
        The path to the geojson file

    Returns
    -------
    List[geojson.feature.Feature]
        A list of features
    """
    with open(fpath, "r") as f:
        data = geojson.load(f)
        if not isinstance(data, list):
            data = [data]
    return data


def get_roi_embeds(
    tile_embeds: Dict[str, torch.Tensor],
    roi_tiles: Set[Tuple[int, int]],
) -> List[torch.Tensor]:
    """
    Returns a list of tile embedding vectors for the tiles contained in
    roi_tiles.

    Parameters
    ----------
    tile_embeds : Dict[str, torch.Tensor]
        The tile embedding dict for a single slide. Must have keys
        "tile_embeds" and "coords" containing tensors with model embeddings
        and tile coordinates

    roi_tiles : Set[Tuple[int, int]]
        The (x, y) coordinate pairs identifying tiles in the ROI to extract
        embeddings for

    Returns
    -------
    List[torch.Tensor]
        A list of tile embedding vectors
    """
    if len(roi_tiles) == 0:
        return []

    idxs = []
    for i, coords in enumerate(tile_embeds["coords"]):
        pair = coords[0].item(), coords[1].item()
        if pair in roi_tiles:
            idxs.append(i)

    roi_embeds = list(itemgetter(*idxs)(tile_embeds["tile_embeds"]))
    return roi_embeds
