from typing import Dict, Tuple

import gigapath.slide_encoder as slide_encoder
import timm
import torch
from torch import nn
from torchvision import transforms

from models.foundation_models.fm import FoundationModel


class GigaPath(FoundationModel):
    def __init__(
        self,
        tiles_dir: str,
        tile_embeds_path: str,
        slide_embeds_path: str,
        global_pool: bool = True,
        **kwargs,
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

        global_pool : bool
            Whether to use global pooling in the slide encoder. If False, the
            output will be the CLS token embedding
        """
        self._global_pool = global_pool
        super().__init__(tiles_dir, tile_embeds_path, slide_embeds_path)

    def _load_tile_encoder(self) -> Tuple[nn.Module, transforms.Compose]:
        """
        Loads the Prov-GigaPath tile encoder and transforms required for tile
        inference.

        Returns
        -------
        Tuple[nn.Module, transforms.Compose]
            The GigaPath tile encoder model and transforms
        """
        model = timm.create_model(
            "hf_hub:prov-gigapath/prov-gigapath", pretrained=True
        )
        transform = transforms.Compose(
            [
                transforms.Resize(
                    256, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )
        return model, transform

    def _load_slide_encoder(self) -> nn.Module:
        model = slide_encoder.create_model(
            "hf_hub:prov-gigapath/prov-gigapath",
            "gigapath_slide_enc12l768d",
            1536,
            global_pool=self._global_pool,
        )
        return model

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
            coords = data["coords"]
            if len(embed.shape) == 2:
                tile_embeds = embed.unsqueeze(0)
                coords = coords.unsqueeze(0)

            # run inference
            with torch.autocast("cuda", dtype=torch.float16):
                slide_embed = self._slide_encoder(
                    tile_embeds.to(device),
                    coords.to(device),
                    all_layer_embed=True,
                )
                outputs = {
                    "layer_{}_embed".format(i): slide_embed[i].detach().cpu()
                    for i in range(len(slide_embed))
                }
                outputs["last_layer_embed"] = slide_embed[-1].detach().cpu()

            # save only the final embedding
            slide_embeds[id] = outputs["last_layer_embed"]

        return slide_embeds
