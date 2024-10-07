import os
import pickle
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision import transforms


class SlideEncodingDataset(Dataset):
    """
    Dataset for input into slide encoders.
    """

    def __init__(
        self, tile_embed_paths: List[str], labels: Dict[str, int]
    ) -> None:
        """
        Parameters
        ----------
        tile_embed_paths : List[str]
            Paths to tile embedding pickle files. Each pickle file should
            contain a a dictionary with keys "tile_embeds" and "coords" and
            values that contain tensors matching the respective keys

        labels : Dict[str, int]
            Integer labels for each slide included in tile_embed_paths. Keys
            are slide ids and values are integer labels
        """
        self.tile_embed_paths = tile_embed_paths
        self.labels = labels

    def __len__(self) -> int:
        """
        Returns the number of slides in the dataset.

        Returns
        -------
        int
            The number of slides in the dataset
        """
        return len(self.tile_embed_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns a tile embeddings for a sampled slide.

        Parameters
        ----------
        idx : int
            The index of the sampled slide in self.tile_embed_paths

        Returns
        -------
        Dict[str, Any]
            Tile embeddings for the sampled slide. Keys are "tile_embeds",
            "coords", "id", and "label"
        """
        embed_path = self.tile_embed_paths[idx]
        embed_name = os.path.basename(embed_path)[:-4]
        specimen_name = embed_name[:6]
        with open(embed_path, "rb") as f:
            embed = pickle.load(f)
        embed["id"] = embed_name
        embed["label"] = self.labels[specimen_name]
        return embed


class TileEncodingDataset(Dataset):
    """
    Taken from prov-gigapath pipeline.

    Dataset for input into tile encoders. Retains image coordinates that are
    encoded in tile image file names.
    """

    def __init__(
        self, image_paths: List[str], transform: transforms.Compose = None
    ) -> None:
        self.transform = transform
        self.image_paths = image_paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path = self.image_paths[idx]
        img_name = os.path.basename(img_path)
        # get x, y coordinates from the image name
        x, y = img_name.split(".png")[0].split("_")
        x, y = int(x.replace("x", "")), int(y.replace("y", ""))
        # load the image
        with open(img_path, "rb") as f:
            img = Image.open(f).convert("RGB")
            if self.transform:
                img = self.transform(img)
        return {
            "img": torch.from_numpy(np.array(img)),
            "coords": torch.from_numpy(np.array([x, y])).float(),
        }


def collate_tile_embeds(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for loading tile embeddings. Each slide has a
    different number of tiles, so padding must be added to smaller tile sets
    to allow batching.

    Parameters
    ----------
    batch : List[Dict[str, Any]]
        A list of samples to be collated into a batch. Each sample is a dict
        containing tensors "tile_embeds" and "coords", and a string "id"

    Returns
    -------
    Dict[str, Any]
        The collated batch with keys matching input keys. "id" is a list of
        slide ids while "tile_embeds" and "coords" remain tensors with an
        added batch dimension
    """
    # get the max length tile "sequence" in the batch
    max_length = 0
    for item in batch:
        max_length = max(max_length, item["tile_embeds"].shape[0])

    # apply padding to tile embeds and coords to ensure all tensors have the
    # same shape prior to collation
    for item in batch:
        tiles = item["tile_embeds"].shape[0]
        lpad = (max_length - tiles) // 2
        rpad = max_length - tiles - lpad
        padding = (0, 0, lpad, rpad)

        item["tile_embeds"] = F.pad(
            item["tile_embeds"],
            padding,
            mode="constant",
        )
        item["coords"] = F.pad(
            item["coords"],
            padding,
            mode="constant",
            value=float("-inf"),
        )

    # collate the tensors by stacking, ids (strings) by creating a list of ids
    collated_batch = {
        k: torch.stack([item[k] for item in batch])
        for k in item.keys()
        if isinstance(item[k], torch.Tensor)
    }
    collated_batch["label"] = torch.tensor([item["label"] for item in batch])
    collated_batch["id"] = [item["id"] for item in batch]
    return collated_batch
