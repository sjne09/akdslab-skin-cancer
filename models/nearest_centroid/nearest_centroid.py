import os
import pickle
from enum import IntEnum
from operator import itemgetter
from random import randint
from typing import Dict, List, Optional, Set, Tuple

import geojson
import matplotlib.pyplot as plt
import numpy as np
import torch
from dotenv import load_dotenv
from shapely import Polygon, box
from sklearn.cluster import KMeans
from torch.nn import functional as F

from data_models.Label import Label
from utils.eval import Evaluator
from utils.load_data import SpecimenData
from utils.slide_utils import load_slide

load_dotenv()
DATA_DIR = os.getenv("DATA_DIR")
OUTPUT_DIR = os.getenv("OUTPUT_DIR")


class NearestCentroid:
    def __init__(
        self, specimen_ids: List[List[str]], embeddings_path: str
    ) -> None:
        self.specimen_ids = specimen_ids
        self.embeddings_path = embeddings_path
        self.centroids: torch.Tensor = None

    def fit(self) -> None:
        """
        Fits the nearest centroid classifier to randomly sampled instances
        from each class.
        """
        # retrieve one randomly sampled specimen from each class
        tile_embeds = self._get_random_specs()

        # set the baseline centroid for the NA class
        na_baseline = tile_embeds["na"]["tile_embeds"].mean(dim=0)

        # fit a kmeans model for each class with 2 centroids (assuming tiles
        # are either class representative or representative of the NA class)
        kmeans = {}
        for c in tile_embeds.keys():
            kmeans[c] = KMeans(n_clusters=2, random_state=0).fit(
                tile_embeds[c]["tile_embeds"]
            )

        class_centroids = []
        for i, kmean in enumerate(kmeans.values()):
            # if the class is NA, use the baseline centroid
            if i == Label.na:
                class_centroids.append(na_baseline)
                continue

            # get the centroid that is furthest from the na baseline centroid
            # assume this is the class representative centroid
            centroids = torch.tensor(
                kmean.cluster_centers_, dtype=torch.float32
            )
            class_rep_centroid = F.cosine_similarity(
                na_baseline, centroids, dim=-1
            ).argmin()
            class_centroids.append(centroids[class_rep_centroid])

        self.centroids = torch.stack(class_centroids, dim=0)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Returns dot products between the rows of X and the columns of
        centroids.
        """
        tile_preds = X @ self.centroids.T
        return tile_preds

    def _get_random_specs(self) -> Dict[str, str]:
        """
        Randomly samples one specimen from each class.

        Returns
        -------
        Dict[str, str]
            Dictionary of tile embeddings for one randomly sampled specimen
            id for each class
        """
        # randomly sample one specimen id from each class
        chosen = {}
        for i in range(len(self.specimen_ids)):
            idx = randint(0, len(self.specimen_ids[i]) - 1)
            chosen[Label(i).name] = self.specimen_ids[i][idx]

        # get the embeddings for the slides for each sampled specimen
        tile_embeds = {
            c.name: {"tile_embeds": [], "coords": []} for c in Label
        }
        for c in Label:
            slides = [
                s
                for s in os.listdir(self.embeddings_path)
                if s[:6] == chosen[c.name]
            ]
            for s in slides:
                with open(os.path.join(self.embeddings_path, s), "rb") as f:
                    data = pickle.load(f)
                    tile_embeds[c.name]["tile_embeds"].append(
                        data["tile_embeds"]
                    )
                    tile_embeds[c.name]["coords"].append(data["coords"])

        # collate the tile embeds for each slide into a single tensor
        tile_embeds = {
            c.name: {
                k: torch.cat(tile_embeds[c.name][k])
                for k in tile_embeds[c.name].keys()
            }
            for c in Label
        }
        return tile_embeds


class AdhocNearestCentroid:
    def __init__(
        self,
        labels_enum: IntEnum,
        centroids: Optional[torch.Tensor] = None,
    ) -> None:
        self.labels = labels_enum
        self.centroids: torch.Tensor = centroids

    def fit(
        self,
        tile_embed_dir: str,
        roi_config: Optional[Dict[str, str]] = None,
        roi_dir: Optional[str] = None,
    ):
        """
        Must provide either roi_config or roi_dir.
        """
        centroids = []
        roi_tiles = {}

        # if roi_config is provided, ROI tiles must be determined
        # otherwise, assume roi_dir is provided with pre-calculated ROI tiles
        if roi_config is not None:
            polygon_dir = roi_config["annotation_directory"]
            slide_dir = roi_config["slide_directory"]

            for name, pg_type in roi_config["polygon_type"].items():
                root = os.path.join(polygon_dir, name)

                # for classes with subclasses, must use modified process
                if pg_type == "subclass":
                    pgs = self._get_subclass_polygons(root)
                    for label, polygons in pgs.items():
                        roi_tiles[label] = self._roi_tiles_by_slide(
                            polygons,
                            slide_dir,
                        )
                else:
                    polygons = self._get_class_polygons(root)
                    roi_tiles[name] = self._roi_tiles_by_slide(pgs, slide_dir)
        else:
            for label in self.labels._member_names_:
                with open(
                    os.path.join(roi_dir, f"{label}-roi.pkl"), "rb"
                ) as f:
                    roi_tiles[label] = pickle.load(f)

        for label in roi_tiles:
            centroids.append(
                self._create_centroid(tile_embed_dir, roi_tiles[label])
            )

        self.centroids = torch.stack(centroids, dim=0)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Returns dot products between the rows of X and the columns of
        centroids.
        """
        if self.centroids is None:
            raise Exception("Model must be fit first.")
        return X @ self.centroids.T

    def save_model(self, output_dir: str):
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
            if polygon.intersects(tile):
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

    def _get_tile_coords(self, slide_path: str) -> List[Tuple[int, int]]:
        tiles = [
            tile[:-4]
            for tile in os.listdir(slide_path)
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
        return (
            int((coords[0] - origin[0]) / downscale_factor),
            int((coords[1] - origin[1]) / downscale_factor),
        )

    def _map_coords(
        self, slide_path: str
    ) -> Dict[Tuple[int, int], Tuple[int, int]]:
        img = load_slide(slide_path)
        coords = self._get_tile_coords(slide_path)
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
        polygon_map: Dict[str, Polygon],
        slide_dir: str,
        classification: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Set[Tuple[int, int]]]:
        relevant_tiles = {}
        for slide_id, polygons in polygon_map.items():
            relevant_tiles[slide_id] = set()
            coord_map = self._map_coords(
                os.path.join(slide_dir), f"{slide_id}.svs"
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
        return torch.stack(cluster, dim=0).mean(dim=0).float()


def plot_results(spec_level_probs, onehot_labels):
    roc_fig, roc_axs = plt.subplots(2, 2, figsize=(10, 10))
    prc_fig, prc_axs = plt.subplots(2, 2, figsize=(10, 10))
    for j, class_of_interest in enumerate(Label._member_names_):
        Evaluator.plot_eval(
            mean_x=np.linspace(0, 1, 100),
            onehot_labels=onehot_labels[:, j],
            probs=spec_level_probs[:, j],
            ax=roc_axs[j // 2][j % 2],
            plot_type="ROC",
            fold_idx=0,
            plot_chance_level=True,
        )

        Evaluator.plot_eval(
            mean_x=np.linspace(0, 1, 100),
            onehot_labels=onehot_labels[:, j],
            probs=spec_level_probs[:, j],
            ax=prc_axs[j // 2][j % 2],
            plot_type="PRC",
            fold_idx=0,
            plot_chance_level=True,
        )

    roc_fig.savefig("outputs/nearest_centroid/mean_pool_roc.png")
    prc_fig.savefig("outputs/nearest_centroid/mean_pool_prc.png")


def main():
    labels_path = os.path.join(DATA_DIR, "labels/labels.csv")
    embeddings_path = os.path.join(OUTPUT_DIR, "prism/tile_embeddings_sorted")

    data = SpecimenData(labels_path)
    clf = NearestCentroid(data.specimens_by_label, embeddings_path)
    clf.fit()

    slides = [
        s for s in os.listdir(embeddings_path) if s[:6] in data.specimens
    ]

    preds = []
    for slide in slides:
        with open(os.path.join(embeddings_path, slide), "rb") as f:
            slide_data = pickle.load(f)
            preds.append(clf.predict(slide_data["tile_embeds"].float()))
    preds = torch.stack(preds)
    ids, spec_level_probs = Evaluator.get_spec_level_probs(
        [s[:-4] for s in slides], preds
    )
    onehot_labels = np.array(itemgetter(*ids)(data.onehot_labels))
    plot_results(spec_level_probs, onehot_labels)


if __name__ == "__main__":
    main()
