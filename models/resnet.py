from torch import nn
from torch.utils.checkpoint import checkpoint
from torchvision.models import resnet18

from .agg import MILClassifier


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


class CheckpointedResNetFeatureExtractor(nn.Module):
    """
    ResNet18 feature extractor for tiles.
    """

    def __init__(self):
        super().__init__()
        tile_encoder = resnet18(weights="DEFAULT")

        # deconstruct model to remove final FC layer and enable checkpointing
        self.stem = nn.Sequential(*list(tile_encoder.children())[:4])
        self.layer1 = tile_encoder.layer1
        self.layer2 = tile_encoder.layer2
        self.layer3 = tile_encoder.layer3
        self.layer4 = tile_encoder.layer4
        self.avgpool = tile_encoder.avgpool

        # embed_dim is the size of output of the final layer
        self.embed_dim = self.layer4[-1].bn2.num_features

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
        x = self.stem(x)  # (S, 64, 56, 56)
        x = checkpoint(self.layer1, x)  # (S, 64, 56, 56)
        x = checkpoint(self.layer2, x)  # (S, 128, 28, 28)
        x = checkpoint(self.layer3, x)  # (S, 256, 14, 14)
        x = checkpoint(self.layer4, x)  # (S, 512, 7, 7)
        x = self.avgpool(x)  # (S, 512, 1, 1)
        return x.squeeze()


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
        tile_embeds = self.tile_encoder(x).unsqueeze(0)  # (1, S, d_embed)
        x = self.slide_classifier(tile_embeds, coords)  # (1, d_out)
        return x


class DistributedResNetWithABMIL(nn.Module):
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
        self.tile_encoder = resnet18(weights="DEFAULT").to("cuda:0")
        embed_dim = self.tile_encoder.fc.out_features
        self.slide_classifier = MILClassifier(
            embed_dim,
            out_features,
            attn_heads,
            gated_attn,
        ).to("cuda:1")

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
        # ensure that the input is on the correct device
        x = x.to("cuda:0")
        tile_embeds = self.tile_encoder(x).unsqueeze(0)  # (1, S, d_embed)

        # move the embeddings and coords to GPU 1 for remaining layers
        tile_embeds = tile_embeds.to("cuda:1")
        coords = coords.to("cuda:1")
        x = self.slide_classifier(tile_embeds, coords)  # (1, d_out)
        return x
