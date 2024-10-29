from torch import nn
from torchvision.models import resnet18

from .agg import MILClassifier


class ResNetWithABMIL(nn.Module):
    """
    ResNet18 combined with ABMIL for tile aggregation, with an MLP
    classification head. For training, the model is designed to take a single
    batch of all tiles for a single slide at a time.
    """

    def __init__(
        self,
        out_features: int,
        attn_heads: int = 1,
        gated_attn: bool = False,
    ) -> None:
        super().__init__()
        self.tile_encoder = resnet18(weights="DEFAULT")
        embed_dim = self.tile_encoder.fc.out_features
        self.slide_classifier = MILClassifier(
            embed_dim,
            out_features,
            attn_heads,
            gated_attn,
        )

    def forward(self, x, coords):
        """
        Parameters
        ----------
        x : torch.Tensor
            (S, C, H, W)

        coords : torch.Tensor
            (S, 2)
        """
        # TODO: consider only using randomly sampled tiles such that batches
        #       can be smaller
        #       this is what renyu did and what was done for UNI (I believe)
        #       but not sure about how best to sample
        print(f"inputs: {x.shape}")
        print(f"coords: {coords.shape}")
        tile_embeds = self.tile_encoder(x).unsqueeze(0)  # (1, S, d_embed)
        print(f"tile_embeds: {tile_embeds.shape}")
        x = self.slide_classifier(tile_embeds, coords)  # (1, d_out)
        print(f"clf: {x.shape}")
        return x
