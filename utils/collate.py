from typing import Any, Dict, List

import torch
from torch.nn import functional as F


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
