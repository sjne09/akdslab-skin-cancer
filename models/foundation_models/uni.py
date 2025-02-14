from typing import Tuple

import timm
import torch
from torch import nn
from torchvision import transforms

from models.foundation_models.fm import FoundationModel


class UNI(FoundationModel):
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
        super().__init__(tiles_dir, tile_embeds_path, slide_embeds_path)

    def _load_tile_encoder(self) -> Tuple[nn.Module, transforms.Compose]:
        """
        Loads the UNI model and transforms required for inference.

        Returns
        -------
        Tuple[nn.Module, transforms.Compose]
            The UNI tile encoder model and transforms
        """
        model: nn.Module = timm.create_model(
            "vit_large_patch16_224",
            img_size=224,
            patch_size=16,
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True,
        )
        model.load_state_dict(
            torch.load(
                "/opt/gpudata/skin-cancer/models/UNI/assets/ckpts/"
                + "vit_large_patch16_224.dinov2.uni_mass100k/"
                + "pytorch_model.bin",
            ),
            strict=True,
        )
        transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )
        return model, transform

    def _load_slide_encoder(self) -> None:
        return None

    def _run_slide_encoder_inference(self, device: torch.device) -> None:
        raise NotImplementedError("Slide encoder not implemented for UNI")
