import os
import pickle
from enum import IntEnum
from operator import itemgetter
from typing import Dict, List, Optional, Set, Tuple

import geojson
import torch
from shapely import Polygon, box, contains, intersects

from utils.slide_utils import load_slide


class AdhocNearestCentroid:
    def __init__(
        self,
        labels_enum: IntEnum,
        centroids: Optional[torch.Tensor] = None,
        mode: str = "intersects",
    ) -> None:
        if mode not in {"intersects", "contains"}:
            raise ValueError("mode must be either 'intersects' or 'contains'")
        self.labels = labels_enum
        self.centroids: torch.Tensor = centroids
        self.mode = intersects if mode == "intersects" else contains

    def fit(
        self,
        tile_embed_dir: str,
        *,
        roi_config: Optional[Dict[str, str]] = None,
        roi_dir: Optional[str] = None,
    ):
        """
        Fit the model. Must provide either roi_config or roi_dir.

        Parameters
        ----------
        tile_embed_dir : str
            The path to the directory containing tile embedding data

        roi_config : Optional[Dict[str, str]]
            The config specifying how to determine ROIs to extract tiles from.
            Must be provided if roi_dir is not provided

        roi_dir : Optional[str]
            Path to directory containing pre-calculated ROI tiles. Must be
            provided if roi_config is not provided. Assumes naming convention
            used in _roi_tiles_by_slide
        """
        centroids = []
        roi_tiles = {}

        assert roi_config is not None or roi_dir is not None

        if roi_config is not None:
            # for each classification specified in polygon_type, get the
            # polygons, then extract the tiles that fall in the polygons
            polygon_dir = roi_config["annotation_directory"]
            slide_dir = roi_config["slide_directory"]
            tiles_dir = roi_config["tiles_directory"]

            for name, pg_type in roi_config["polygon_type"].items():
                root = os.path.join(polygon_dir, name)
                # for classes with subclasses, must use modified process
                if pg_type == "subclass":
                    pgs = self._get_subclass_polygons(root)
                    for label, polygons in pgs.items():
                        roi_tiles[label] = self._roi_tiles_by_slide(
                            polygon_map=polygons,
                            slide_dir=slide_dir,
                            tiles_dir=tiles_dir,
                            classification=label,
                            output_dir=roi_config["roi_tiles_output_directory"]
                            or None,
                        )
                else:
                    polygons = self._get_class_polygons(root)
                    roi_tiles[name] = self._roi_tiles_by_slide(
                        polygon_map=polygons,
                        slide_dir=slide_dir,
                        tiles_dir=tiles_dir,
                        classification=name,
                        output_dir=roi_config["roi_tiles_output_directory"]
                        or None,
                    )
        else:
            for label in self.labels._member_names_:
                with open(
                    os.path.join(roi_dir, f"{label}-roi.pkl"), "rb"
                ) as f:
                    roi_tiles[label] = pickle.load(f)

        for label in roi_tiles:
            centroid = self._create_centroid(tile_embed_dir, roi_tiles[label])
            if centroid is not None:
                centroids.append(centroid)

        self.centroids = torch.stack(centroids, dim=0)

    def predict(
        self, X: torch.Tensor, mode: str = "dot_product"
    ) -> torch.Tensor:
        """
        Returns predictions as logits based on the selected mode.

        Parameters
        ----------
        X : torch.Tensor
            The tiles to run inference on; shape (N, embed_dim)

        mode : str
            The mode to use for determining centroid distance/similarity.
            Options are:
                "dot_product": the raw dot products of the rows of X and the
                centroids
                "euclidean": the euclidean distance between the rows of X and
                the centroids
                "cosine": the cosine similarity between rows of X and the
                centroids

        Returns
        -------
        torch.Tensor
            The model outputs; shape (N, C), where C is the number of classes/
            centroids
        """
        if self.centroids is None:
            raise Exception("Model must be fit first.")

        modes = {
            "dot_product": self._dot_product,
            "euclidean": self._euclidean_distance,
            "cosine": self._cosine_similarity,
        }

        try:
            return modes[mode](X)
        except KeyError:
            raise ValueError("Invalid mode selected.")

    def save_model(self, output_dir: str) -> None:
        """
        Pickles the model centroids.

        Parameters
        ----------
        output_dir : str
            The directory to save the centroids to
        """
        with open(os.path.join(output_dir, "param.pkl"), "wb") as f:
            pickle.dump(self.centroids, f)

    def _extract_relevant_tiles(
        self,
        polygon: Polygon,
        coords: List[Tuple[int, int]],
        tile_size: int = 256,
    ) -> Set[Tuple[int, int]]:
        """
        Determines which tiles from the list of tile coordinates intersect the
        polygon and returns the list of relevant tiles as a set.

        Parameters
        ----------
        polygon : shapely.Polygon
            The polygon to use as a bounding box

        coords : List[Tuple[int, int]]
            A list containing (x, y) coordinate tuples

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
            if self.mode(polygon, tile):
                relevant_tiles.append((xmin, ymin))

        return set(relevant_tiles)

    def _get_subclass_polygons(
        self, root: str
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
            features = self._get_geojson_features(fpath)

            # for each shape in the geojson, extract the tissue type and add
            # the polygon to the corresponding entry in geoms
            for shape in features:
                tissue_type = shape["properties"]["name"]
                polygon = Polygon(shape["geometry"]["coordinates"][0])

                if tissue_type not in polygons:
                    polygons[tissue_type] = {slide_name: [polygon]}
                elif slide_name not in polygons[tissue_type]:
                    polygons[tissue_type][slide_name] = [polygon]
                else:
                    polygons[tissue_type][slide_name].append(polygon)
        return polygons

    def _get_class_polygons(self, root: str) -> Dict[str, List[Polygon]]:
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
            features = self._get_geojson_features(fpath)

            # for each shape in the geojson, extract the tissue type and add
            #  the polygon to the corresponding entry in geoms
            geoms = []
            for shape in features:
                polygon = Polygon(shape["geometry"]["coordinates"][0])
                geoms.append(polygon)
            polygons[slide_name] = geoms

        return polygons

    def _get_geojson_features(
        self, fpath: str
    ) -> List[geojson.feature.Feature]:
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

    def _get_tile_coords(self, tiles_path: str) -> List[Tuple[int, int]]:
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
            tile[:-4]
            for tile in os.listdir(tiles_path)
            if tile.endswith(".png")
        ]

        coords = []
        for tile in tiles:
            x, y = tile.split("_")
            x, y = int(x.replace("x", "")), int(y.replace("y", ""))
            coords.append((x, y))
        return coords

    def _calculate_original_coords(
        self,
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

    def _map_coords(
        self, slide_path: str, tiles_path: str
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
        coords = self._get_tile_coords(tiles_path)
        # map the original coords to the modified coords
        coord_mapping = {
            self._calculate_original_coords(
                pair, img["origin"], img["scale"]
            ): pair
            for pair in coords
        }
        return coord_mapping

    def _roi_tiles_by_slide(
        self,
        polygon_map: Dict[str, List[Polygon]],
        slide_dir: str,
        tiles_dir: str,
        classification: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Set[Tuple[int, int]]]:
        """
        Gets relevant tiles intersecting each polygon and returns a mapping
        of slide_ids to relevant tile coords.

        Parameters
        ----------
        polygon_map : Dict[str, List[Polygon]]
            A dict with slide ids as keys and a list of polygons as values

        slide_dir : str
            The path to the directory containing slide .svs images

        tiles_dir : str
            The path to the directory containing subdirectories with tile
            images for each slide

        classification : Optional[str]
            The classification for the extracted tiles, used for naming the
            pickled results if output_dir is provided

        output_dir : Optional[str]
            The directory to save the output dict to

        Returns
        -------
        Dict[str, Set[Tuple[int, int]]]
            A dict with slide ids as keys and a set of tuples containing
            (x, y) coordinate pairs as values
        """
        relevant_tiles = {}
        for slide_id, polygons in polygon_map.items():
            relevant_tiles[slide_id] = set()
            coord_map = self._map_coords(
                os.path.join(slide_dir, f"{slide_id}.svs"),
                os.path.join(tiles_dir, f"{slide_id}.svs"),
            )
            for polygon in polygons:
                roi_tiles = self._extract_relevant_tiles(
                    polygon, coord_map.keys()
                )
                if len(roi_tiles) > 0:
                    roi_tiles = set(itemgetter(*roi_tiles)(coord_map))
                    relevant_tiles[slide_id].update(roi_tiles)

        if output_dir is not None:
            with open(
                os.path.join(output_dir, f"{classification}-roi.pkl"), "wb"
            ) as f:
                pickle.dump(relevant_tiles, f)

        return relevant_tiles

    def _get_roi_embeds(
        self,
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

    def _create_centroid(
        self, tile_embeds_path: str, roi_tiles: Dict[str, Set[Tuple[int, int]]]
    ) -> torch.Tensor:
        """
        Create a centroid vector from the embeddings for the tiles with coords
        contained in roi_tiles. Centroid is calculated using global average
        pooling (GAP).

        Parameters
        ----------
        tile_embeds_path : str
            The path to the tile embeds for all slides

        roi_tiles : Dict[str, Set[Tuple[int, int]]]
            A dict containing tile coords for ROIs corresponding to a single
            label. Each entry in the dict should contain a slide id as key and
            a set of (x, y) tuples as values

        Returns
        -------
        torch.Tensor
            The centroid vector
        """
        cluster = []
        for slide in roi_tiles:
            with open(
                os.path.join(tile_embeds_path, f"{slide}.pkl"), "rb"
            ) as f:
                embeds = pickle.load(f)
            cluster.extend(self._get_roi_embeds(embeds, roi_tiles[slide]))

        if cluster:
            return torch.stack(cluster, dim=0).mean(dim=0).float()
        return None

    def _dot_product(self, X: torch.Tensor) -> torch.Tensor:
        return X @ self.centroids.T

    def _cosine_similarity(self, X: torch.Tensor) -> torch.Tensor:
        dp = self._dot_product(X)
        X_norm = torch.norm(X, dim=-1, keepdim=True)
        centroid_norm = torch.norm(self.centroids, dim=-1, keepdim=True)
        return dp / (X_norm @ centroid_norm.T)

    def _euclidean_distance(self, X: torch.Tensor) -> torch.Tensor:
        # reshape to enable broadcasting
        X_expanded = X.unsqueeze(1)
        centroids_expanded = self.centroids.unsqueeze(0)

        # calculate euclidean distance
        squared_diff = (X_expanded - centroids_expanded) ** 2
        sum_squared_diff = squared_diff.sum(dim=-1)
        return torch.sqrt(sum_squared_diff)

    def load_model(self, path: str) -> None:
        """
        Load pickled centroids.

        Parameters
        ----------
        path : str
            Path to pickled centroids
        """
        with open(path, "rb") as f:
            self.centroids = pickle.load(f)
