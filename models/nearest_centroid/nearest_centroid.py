import os
import pickle
from enum import IntEnum
from operator import itemgetter
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
from shapely import Polygon, contains, intersects

from .coords import map_coords
from .roi import (
    extract_relevant_tiles,
    get_class_polygons,
    get_roi_embeds,
    get_subclass_polygons,
)


class NearestCentroid:
    def __init__(
        self,
        labels_enum: IntEnum,
        centroids: Optional[Union[torch.Tensor, os.PathLike, str]] = None,
        mode: str = "intersects",
    ) -> None:
        if mode not in {"intersects", "contains"}:
            raise ValueError("mode must be either 'intersects' or 'contains'")
        self.labels = labels_enum

        if isinstance(centroids, os.PathLike):
            self.load_model(centroids)
        else:
            self.centroids: torch.Tensor = centroids

        self.mode = intersects if mode == "intersects" else contains

        # tile coords for tiles within ROIs for each slide used for centroids
        self.roi_tiles: Dict[str, Set[Tuple[int, int]]] = None

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
                    pgs = get_subclass_polygons(root)
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
                    polygons = get_class_polygons(root)
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

        for label in self.labels._member_names_:
            centroid = self._create_centroid(tile_embed_dir, roi_tiles[label])
            if centroid is not None:
                centroids.append(centroid)

        self.centroids = torch.stack(centroids, dim=0)
        self.roi_tiles = roi_tiles

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
                "modified_dot": the negative absolute value of the dot
                products of the rows of X and the centroids less the squared
                of the centroids
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
            "modified_dot": self._modified_dot_product,
            "euclidean": self._euclidean_distance,
            "cosine": self._cosine_similarity,
        }

        try:
            return modes[mode](X)
        except KeyError:
            raise ValueError("Invalid mode selected.")

    def save_model(self, fpath: str) -> None:
        """
        Pickles the model centroids.

        Parameters
        ----------
        fpath : str
            The file to save the centroids to
        """
        with open(fpath, "wb") as f:
            pickle.dump(self.centroids, f)

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
            coord_map = map_coords(
                os.path.join(slide_dir, f"{slide_id}.svs"),
                os.path.join(tiles_dir, f"{slide_id}.svs"),
            )
            for i, polygon in enumerate(polygons):
                roi_tiles = extract_relevant_tiles(
                    polygon, coord_map.keys(), self.mode
                )
                if len(roi_tiles) > 0:
                    modified_coords = itemgetter(*roi_tiles)(coord_map)
                    if isinstance(modified_coords, tuple) and isinstance(
                        modified_coords[0], int
                    ):
                        roi_tiles = {modified_coords}
                    else:
                        roi_tiles = set(modified_coords)
                    relevant_tiles[slide_id].update(roi_tiles)

        if output_dir is not None:
            with open(
                os.path.join(output_dir, f"{classification}-roi.pkl"), "wb"
            ) as f:
                pickle.dump(relevant_tiles, f)

        return relevant_tiles

    @staticmethod
    def _create_centroid(
        tile_embeds_path: str, roi_tiles: Dict[str, Set[Tuple[int, int]]]
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
            cluster.extend(get_roi_embeds(embeds, roi_tiles[slide]))

        if cluster:
            return torch.stack(cluster, dim=0).float().mean(dim=0)
        return None

    def _dot_product(self, X: torch.Tensor) -> torch.Tensor:
        """
        Calculates the dot product between each row of X and each of
        the model's centroids.

        Parameters
        ----------
        X : torch.Tensor
            A 2-dimensional tensor of shape [N, d_model] where d_model
            is the dimension of the centroid vectors

        Returns
        -------
        torch.Tensor
            A 2-dimensional tensor of shape [N, C] containing the dot
            products between the rows of X and each of the centroids
        """
        return X @ self.centroids.T

    def _modified_dot_product(self, X: torch.Tensor) -> torch.Tensor:
        dp = self._dot_product(X)
        centroids_dot = torch.diagonal(self._dot_product(self.centroids))
        return -(dp - centroids_dot).abs()

    def _cosine_similarity(self, X: torch.Tensor) -> torch.Tensor:
        """
        Calculates the cosine similarity between each row of X and each
        of the model's centroids.

        Parameters
        ----------
        X : torch.Tensor
            A 2-dimensional tensor of shape [N, d_model] where d_model
            is the dimension of the centroid vectors

        Returns
        -------
        torch.Tensor
            A 2-dimensional tensor of shape [N, C] containing the cos
            similarities between the rows of X and each of the
            centroids
        """
        dp = self._dot_product(X)
        X_norm = torch.norm(X, dim=-1, keepdim=True)
        centroid_norm = torch.norm(self.centroids, dim=-1, keepdim=True)
        return dp / (X_norm @ centroid_norm.T)

    def _euclidean_distance(self, X: torch.Tensor) -> torch.Tensor:
        """
        Calculates the euclidean distance between each row of X and each
        of the model's centroids.

        Parameters
        ----------
        X : torch.Tensor
            A 2-dimensional tensor of shape [N, d_model] where d_model
            is the dimension of the centroid vectors

        Returns
        -------
        torch.Tensor
            A 2-dimensional tensor of shape [N, C] containing the
            euclidean distances between the rows of X and each of the
            centroids
        """
        # reshape to enable broadcasting
        X_expanded = X.unsqueeze(1)
        centroids_expanded = self.centroids.unsqueeze(0)

        # calculate euclidean distance
        squared_diff = (X_expanded - centroids_expanded) ** 2
        sum_squared_diff = squared_diff.sum(dim=-1)
        return torch.sqrt(sum_squared_diff)
