import os
import pickle
from typing import Dict, List

import torch
from torch.utils.data import Dataset


class SlideEncodingDataset(Dataset):
    """
    Dataset for input into slide encoders.
    """

    def __init__(
        self, tile_embed_paths: List[str], labels: Dict[str, int]
    ) -> None:
        self.tile_embed_paths = tile_embed_paths
        self.labels = labels

    def __len__(self) -> int:
        return len(self.tile_embed_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        embed_path = self.tile_embed_paths[idx]
        embed_name = os.path.basename(embed_path)[:-4]
        specimen_name = embed_name[:6]
        with open(embed_path, "rb") as f:
            embed = pickle.load(f)
        embed["id"] = embed_name
        embed["label"] = self.labels[specimen_name]
        return embed
