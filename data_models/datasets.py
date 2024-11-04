import os
import pickle
from operator import itemgetter
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from utils.tile_embed_postproc import add_positions


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
        embed_name = os.path.basename(embed_path)[:-4]  # trim extension
        specimen_name = embed_name[:6]
        with open(embed_path, "rb") as f:
            embed = pickle.load(f)
        embed["id"] = embed_name
        embed["label"] = self.labels[specimen_name]
        return embed


class SlideClassificationDataset(Dataset):
    """
    Dataset for input into slide classifiers.
    """

    def __init__(
        self,
        slide_embeds_path: str,
        slide_ids: List[str],
        labels: Dict[str, int],
    ) -> None:
        """
        Parameters
        ----------
        slide_embeds_path : str
            Path to slide embedding pickle file. The file should contain a
            dictionary with slide ids as keys and slide embedding tensors as
            values

        slide_ids : List[str]
            The slide ids to include in the dataset

        labels : Dict[str, int]
            Integer labels for each slide included in the dataset. Keys are
            slide ids and values are integer labels
        """
        self.slide_ids = slide_ids
        with open(slide_embeds_path, "rb") as f:
            slide_embeds = pickle.load(f)
            self.slide_embeds = itemgetter(*slide_ids)(slide_embeds)
        self.labels = labels

    def __len__(self) -> int:
        """
        Returns the number of slides in the dataset.

        Returns
        -------
        int
            The number of slides in the dataset
        """
        return len(self.slide_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns a slide embedding for a sampled slide.

        Parameters
        ----------
        idx : int
            The index of the sampled slide in self.slide_names

        Returns
        -------
        Dict[str, Any]
            Slide embedding for the sampled slide. Keys are "slide_embed",
            "id", and "label"
        """
        embed = {}
        embed["slide_embed"] = self.slide_embeds[idx].view(-1)
        embed_name = self.slide_ids[idx]
        specimen_name = embed_name[:6]
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


class TileLoaderDataset(Dataset):
    def __init__(
        self,
        tile_dirs: List[str],
        labels: Dict[str, int],
        batch_size: int,
        random_samples: int,
        transform: transforms.Compose,
    ):
        self.tile_dirs = tile_dirs
        self.labels = labels
        self.batch_size = batch_size
        self.random_samples = random_samples
        self.transform = transform

    def __len__(self):
        return len(self.tile_dirs)

    def __getitem__(self, idx):
        image_paths = [
            os.path.join(self.tile_dirs[idx], img)
            for img in os.listdir(self.tile_dirs[idx])
            if img.endswith(".png")
        ]
        dataset = TileEncodingDataset(image_paths, self.transform)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=SubsetRandomSampler(
                list(range(len(dataset))), self.random_samples
            ),
        )
        label = torch.tensor(
            [self.labels[os.path.basename(self.tile_dirs[idx])[:6]]]
        )
        return (loader, label)


class ResNetDataset(Dataset):
    """
    Taken from prov-gigapath pipeline.

    Dataset for input into tile encoders. Retains image coordinates that are
    encoded in tile image file names.
    """

    def __init__(
        self,
        tile_dir: str,
        slide_ids: List[str],
        labels: Dict[str, int],
        transform: transforms.Compose = None,
    ) -> None:
        self.tile_dir = tile_dir
        self.slide_ids = slide_ids
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.slide_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        slide_id = self.slide_ids[idx]
        specimen_id = slide_id[:6]
        slide_dir = os.path.join(self.tile_dir, slide_id)

        slide_data = []
        for tile in os.listdir(slide_dir):
            if tile.endswith(".png"):
                img_path = os.path.join(slide_dir, tile)
                img_name = os.path.basename(img_path)
                # get x, y coordinates from the image name
                x, y = img_name.split(".png")[0].split("_")
                x, y = int(x.replace("x", "")), int(y.replace("y", ""))
                # load the image
                with open(img_path, "rb") as f:
                    img = Image.open(f).convert("RGB")
                    if self.transform:
                        img = self.transform(img)

                slide_data.append(
                    {
                        "img": torch.from_numpy(np.array(img)),
                        "coords": torch.from_numpy(np.array([x, y])).float(),
                    }
                )
        collated_tiles = collate_tiles(slide_data)
        collated_tiles["label"] = torch.tensor([self.labels[specimen_id]])
        return collated_tiles


class SubsetRandomSampler(torch.utils.data.Sampler):
    def __init__(self, indices: List[int], length: int) -> None:
        self.indices = indices
        self.length = length

    def __iter__(self):
        for i in torch.randperm(len(self.indices))[
            : min(self.length, len(self.indices))
        ]:
            yield self.indices[i]

    def __len__(self):
        return self.length


def collate_tiles(
    batch: List[Dict[str, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for loading tiles for use with ResNet18 + ABMIL.

    Parameters
    ----------
    batch : List[Dict[str, torch.Tensor]]
        A list of all tiles for a slide to be collated into a batch. Each
        tile's data is a dict containing tensors "img", "coords", and "label"

    Returns
    -------
    Dict[str, torch.Tensor]
        The collated batch with keys "img", "coords", and "pos". "img" and
        "coords" are just the collated inputs, "pos" is a tensor of shape
        (B, 2) with the relative position data of each tile based on coords
    """
    collated_batch = {
        k: torch.stack([item[k] for item in batch]) for k in batch[0].keys()
    }
    add_positions(collated_batch)
    return collated_batch


def collate_tile_embeds(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for loading tile embeddings. Each slide has a
    different number of tiles, so padding must be added to smaller tile sets
    to allow batching.

    Parameters
    ----------
    batch : List[Dict[str, Any]]
        A list of samples to be collated into a batch. Each sample is a dict
        containing tensors "tile_embeds", "coords", and "pos"; a string "id",
        and an int "label"

    Returns
    -------
    Dict[str, Any]
        The collated batch with keys matching input keys. "id" is a list of
        slide ids, "label" is a 1D tensor containing labels, and
        "tile_embeds", "coords", and "pos" remain tensors with an added
        batch dimension
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
        item["pos"] = F.pad(
            item["pos"], padding, mode="constant", value=float("-inf")
        )

    collated_batch = {
        k: torch.stack([item[k] for item in batch])
        for k in ["tile_embeds", "coords", "pos"]
    }
    collated_batch["label"] = torch.tensor([item["label"] for item in batch])
    collated_batch["id"] = [item["id"] for item in batch]
    return collated_batch


def collate_slide_embeds(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for loading slide embeddings.

    Parameters
    ----------
    batch : List[Dict[str, Any]]
        A list of samples to be collated into a batch. Each sample is a dict
        containing a tensor "slide_embed", an int "label", and a string "id"

    Returns
    -------
    Dict[str, Any]
        The collated batch with keys matching input keys. "id" is a list of
        slide ids, "label" is a 1D tensor containing labels, and
        "slide_embed" remains a tensor with an added batch dimension
    """
    collated_batch = {}
    collated_batch["slide_embed"] = torch.stack(
        [item["slide_embed"] for item in batch]
    )
    collated_batch["label"] = torch.tensor([item["label"] for item in batch])
    collated_batch["id"] = [item["id"] for item in batch]
    return collated_batch
