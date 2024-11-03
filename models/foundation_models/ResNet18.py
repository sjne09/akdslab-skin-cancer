from typing import Tuple

import torch
from torch import nn
from torchvision import transforms

from models.foundation_models.FoundationModel import FoundationModel
from models.resnet import ResNetFeatureExtractor


class ResNet18(FoundationModel):
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
        Loads the resnet18 model and transforms required for inference.

        Returns
        -------
        Tuple[nn.Module, transforms.Compose]
            The resnet18 model and transforms
        """
        model = ResNetFeatureExtractor()
        transform = transforms.Compose(
            [
                transforms.Resize(
                    256, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.CenterCrop(224),
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
        raise NotImplementedError("Slide encoder not implemented for resnet")
