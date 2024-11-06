import os
import pickle
from operator import itemgetter
from random import randint
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from torch.nn import functional as F

from data_models.Label import Label
from utils.eval import Evaluator
from utils.load_data import SpecimenData

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

        # stack and normalize to reduce computation required for prediction
        class_centroids = torch.stack(class_centroids, dim=0)
        self.centroids = F.normalize(
            class_centroids, dim=-1
        )  # (n_classes, n_features)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predicts the class of the input data based on the nearest centroid
        """
        # normalize X and compute the dot product with the centroids to get
        # the cosine similarities
        X = F.normalize(X, dim=-1)
        tile_preds = X @ self.centroids.T
        # return torch.max(tile_preds, dim=0).values.softmax(dim=-1)
        return torch.mean(tile_preds, dim=0).softmax(dim=-1)

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
