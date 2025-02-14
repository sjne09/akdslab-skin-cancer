import os
import pickle
from abc import ABC, abstractmethod
from typing import Dict, Iterator, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data_models.datasets import TileEncodingDataset


class FoundationModel(ABC):
    def __init__(
        self, tiles_dir: str, tile_embeds_path: str, slide_embeds_path: str
    ) -> None:
        """
        Parameters
        ----------
        tiles_dir : str
            The path to the directory containing the image tiles

        tile_embeds_path : str
            The path to the tile embeddings/where to save the tile embeddings

        slide_embeds_path : str
            The path to the slide embeddings/where to save the slide embeddings
        """
        self._tile_encoder, self._transform = self._load_tile_encoder()
        self._slide_encoder = self._load_slide_encoder()
        self._tiles_dir = tiles_dir
        self._tile_embeds_path = tile_embeds_path
        self._slide_embeds_path = slide_embeds_path

    @abstractmethod
    def _load_tile_encoder(
        self,
    ) -> Tuple[torch.nn.Module, transforms.Compose]:
        """Loads the tile encoder model and transforms"""
        pass

    @abstractmethod
    def _load_slide_encoder(self) -> Optional[torch.nn.Module]:
        """Loads the slide encoder model"""
        pass

    def _run_tile_encoder_inference(
        self,
        tile_paths: List[str],
        batch_size: int,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """
        Runs inference using a tile encoding model.

        Parameters
        ----------
        tile_paths : List[str]
            A list of paths to tiles of a single WSI

        batch_size : int
            Batch size to use

        device : torch.device
            The device to send the model and images to

        Returns
        -------
        Dict[str, torch.Tensor]
            A dict with keys "tile_embeds" and "coords" and tensor values with
            shapes (N, D) and (N, 2) where N is the number of tiles for the
            slide and D is the output dimension from the model
        """
        loader = DataLoader(
            TileEncodingDataset(tile_paths, transform=self._transform),
            batch_size=batch_size,
            shuffle=False,
        )

        self._tile_encoder.to(device)
        self._tile_encoder.eval()
        collated_outputs = {"tile_embeds": [], "coords": []}
        with torch.inference_mode():
            for batch in loader:
                output = (
                    self._tile_encoder(batch["img"].to(device)).detach().cpu()
                )
                if len(output.shape) == 1:
                    output = output.unsqueeze(0)
                collated_outputs["tile_embeds"].append(output)

                if len(batch["coords"].shape) == 1:
                    batch["coords"] = batch["coords"].unsqueeze(0)
                collated_outputs["coords"].append(batch["coords"])

        return {k: torch.cat(v) for k, v in collated_outputs.items()}

    @abstractmethod
    def _run_slide_encoder_inference(
        self, device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """
        Runs inference using a slide encoder model.

        Parameters
        ----------
        device : torch.device
            The device to use

        Returns
        -------
        Dict[str, torch.Tensor]
            A dict with keys as slide ids and values as the embedding tensors
        """
        pass

    def create_tile_embeds(
        self,
        device: torch.device,
        batch_size: int = 128,
    ) -> None:
        """
        Create tile embeddings for each slide in the provided tiles directory
        and save to disk. Embeddings will be saved as pickle files. Each
        pickle file will contain the collated tile embeddings for a single
        slide in a dict with keys "tile_embeds" and "coords" and tensor
        values with shapes (N, D) and (N, 2) where N is the number of tiles
        for the slide and D is the output dimension from the model

        Parameters
        ----------
        device : torch.device
            The device to use

        batch_size : int
            Batch size to use
        """
        for dirname in os.listdir(self._tiles_dir):
            dpath = os.path.join(self._tiles_dir, dirname)
            if os.path.isdir(dpath):
                tiles = [
                    os.path.join(dpath, fname)
                    for fname in os.listdir(dpath)
                    if fname.endswith(".png")
                ]
                if len(tiles) > 0:
                    print(f"running for {dpath} with {len(tiles)} tiles")

                    # inference method will move model and tiles to cuda
                    # device
                    # model return includes tensor containing embeddings and
                    # tensor containing tile coords
                    embeds = (
                        dirname,
                        self._run_tile_encoder_inference(
                            tiles,
                            batch_size,
                            device,
                        ),
                    )

                    # pickle the embeddings
                    with open(
                        os.path.join(
                            self._tile_embeds_path,
                            "{name}.pkl".format(
                                name=os.path.splitext(dirname)[0]
                            ),
                        ),
                        "wb",
                    ) as f:
                        pickle.dump(embeds[1], f)

    def create_slide_embeds(self, fname: str, device: torch.device) -> None:
        """
        Creates slide embeddings and save to disk. Embeddings will be saved as
        a pickle file. The pickle file will contain the collated slide
        embeddings for all slides in a dict with keys as slide ids and
        values as the embedding tensors.

        Parameters
        ----------
        fname : str
            The filename to save the slide embeddings to in the slide embeds
            path

        device : torch.device
            The device to use
        """
        embeds_dict = self._run_slide_encoder_inference(device)
        with open(
            os.path.join(self._slide_embeds_path, fname + ".pkl"),
            "wb",
        ) as f:
            pickle.dump(embeds_dict, f)

    def load_tile_embeds(
        self,
    ) -> Iterator[Tuple[str, Dict[str, torch.Tensor]]]:
        """
        Yields tile embeddings for a single slide at a time by accessing
        pickled tile embeddings.

        Yields
        ------
        Tuple[str, Dict[str, torch.Tensor]]
            A tuple containing the slide id and the embedding dict with keys
            "tile_embeds" and "coords"
        """
        for slide in os.listdir(self._tile_embeds_path):
            name = os.path.splitext(slide)[0]
            with open(os.path.join(self._tile_embeds_path, slide), "rb") as f:
                # each .pkl file contains a dict with keys "tile_embeds",
                # "coords" and values torch.Tensor
                emb = pickle.load(f)
                yield (name, emb)

    def create_pooled_slide_embeds(
        self, fname: str, z_norm: bool = False
    ) -> None:
        """
        Creates slide embeddings using the global pooling strategy (i.e., by
        averaging all of the tile embeddings) and saves to disk. Embeddings
        will be saved as a pickle file. The pickle file will contain the
        collated slide embeddings for all slides in a dict with keys as slide
        ids and values as the embedding tensors.

        Parameters
        ----------
        fname : str
            The filename to save the slide embeddings to in the slide embeds
            path

        z_norm : bool
            Whether to z-normalize the slide embeddings
        """
        slide_embeds = {}
        for name, emb in self.load_tile_embeds():
            slide_embeds[name] = emb["tile_embeds"].mean(dim=0)
            if z_norm:
                slide_embeds[name][emb] = (
                    slide_embeds[name][emb] - slide_embeds[name][emb].mean()
                ) / slide_embeds[name][emb].std()

        with open(
            os.path.join(self._slide_embeds_path, fname + ".pkl"),
            "wb",
        ) as f:
            pickle.dump(slide_embeds, f)
