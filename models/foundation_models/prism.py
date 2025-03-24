from typing import Dict, List, Tuple

import timm
import torch
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoModel

from data_processing.datasets import TileEncodingDataset
from models.foundation_models.fm import FoundationModel


class PRISM(FoundationModel):
    def __init__(
        self,
        tiles_dir: str,
        tile_embeds_path: str,
        slide_embeds_path: str,
        **kwargs
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
        super().__init__(tiles_dir, tile_embeds_path, slide_embeds_path)

    def _load_tile_encoder(self) -> Tuple[nn.Module, transforms.Compose]:
        """
        Loads the Virchow v1 model and transforms required for tile inference.

        Returns
        -------
        Tuple[nn.Module, transforms.Compose]
            The virchow tile encoder model and transforms
        """
        # need to specify MLP layer and activation function for proper init
        model = timm.create_model(
            "hf-hub:paige-ai/Virchow",
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )
        transform = create_transform(
            **resolve_data_config(model.pretrained_cfg, model=model)
        )
        return model, transform

    def _load_slide_encoder(self) -> nn.Module:
        """
        Loads the PRISM model for slide inference.

        Returns
        -------
        nn.Module
            The PRISM model, including slide encoder (perceiver) and language
            model
        """
        model = AutoModel.from_pretrained(
            "paige-ai/Prism", trust_remote_code=True
        )
        return model

    def _run_tile_encoder_inference(
        self,
        tile_paths: List[str],
        batch_size: int,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """
        Runs inference using a tile embedding model.

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
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.float16
        ):
            for batch in loader:
                output = (
                    self._tile_encoder(batch["img"].to(device)).detach().cpu()
                )
                if len(output.shape) == 1:
                    output = output.unsqueeze(0)

                # extract the CLS token and patch tokens from the output, then
                # concatenate CLS and the mean of the patch tokens to get the
                # final tile embedding
                class_token = output[:, 0]
                patch_tokens = output[:, 1:]
                embedding = torch.cat(
                    [class_token, patch_tokens.mean(1)], dim=-1
                )
                collated_outputs["tile_embeds"].append(embedding)
                collated_outputs["coords"].append(batch["coords"])

        return {k: torch.cat(v) for k, v in collated_outputs.items()}

    def _run_slide_encoder_inference(
        self, device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """
        Run inference using the slide encoder.

        Parameters
        ----------
        device : torch.device
            The device to use

        Returns
        -------
        Dict[str, torch.Tensor]
            The slide embeddings in a dict with keys as the slide IDs and
            values as the embeddings
        """
        tile_embeds = self.load_tile_embeds()
        self._slide_encoder.to(device)
        self._slide_encoder.eval()

        slide_embeds = {}
        for id, data in tile_embeds:
            embed = data["tile_embeds"]

            with torch.autocast("cuda", torch.float16), torch.inference_mode():
                reprs = self._slide_encoder.slide_representations(
                    embed.unsqueeze(0).to(device)
                )
                slide_embed = reprs["image_embedding"].detach().cpu()

            slide_embeds[id] = slide_embed

        return slide_embeds
