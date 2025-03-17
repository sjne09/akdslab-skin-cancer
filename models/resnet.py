from torch import nn
from torchvision.models import resnet18


class ResNetFeatureExtractor(nn.Module):
    """
    ResNet18 feature extractor for tiles.
    """

    def __init__(self):
        super().__init__()
        tile_encoder = resnet18(weights="DEFAULT")

        # deconstruct model to remove final FC layer
        self.tile_encoder = nn.Sequential(*list(tile_encoder.children())[:-1])

        # embed_dim is the size of output of the final layer
        self.embed_dim = 512

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            (S, 3, 224, 224)

        Returns
        -------
        torch.Tensor
            (S, d_embed)
        """
        x = self.tile_encoder(x)  # (S, 512, 1, 1)
        return x.squeeze()
