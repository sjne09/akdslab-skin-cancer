from typing import List, Optional, Union

import torch
from torch import nn

from .abmil import GatedMILAttention, MILAttention
from .mlp import MLP
from .pos_embedding import PositionalEmbedding


class ABMILWrapper(nn.Module):
    """
    A wrapper for ABMIL to use with nn.Sequential
    """

    def __init__(self, abmil: Union[GatedMILAttention, MILAttention]) -> None:
        super(ABMILWrapper, self).__init__()
        self.abmil = abmil

    def forward(self, x) -> torch.Tensor:
        # only return the output, not the attention weights
        return self.abmil(x)[0]


class EnsembleClassifier(nn.Module):
    def __init__(
        self,
        tile_encoder_dim: Optional[Union[int, List[int]]] = None,
        slide_encoder_dim: Optional[Union[int, List[int]]] = None,
        out_features: int = 4,
        abmil_heads: int = 1,
        gated_abmil: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        tile_encoder_dim : Union[int, List[int]]
            The dimension of outputs from the tile encoder models included in
            the ensemble

        slide_encoder_dim : Union[int, List[int]]
            The dimension of outputs from the slide encoder models included in
            the ensemble

        out_features : int
            The number of features in the output

        abmil_heads : int
            The number of ABMIL attention heads

        gated_abmil : bool
            Whether to use the gated ABMIL architecture
        """
        super().__init__()
        self.total_dim = 0

        # set up tile aggregators if necessary to get slide embeddings
        if tile_encoder_dim is not None:
            self.tile_encoder_dim = (
                tile_encoder_dim
                if isinstance(tile_encoder_dim, list)
                else [tile_encoder_dim]
            )
            self.total_dim += sum(self.tile_encoder_dim)

            # set up ABMIL for tile encoder models
            abmil = MILAttention if not gated_abmil else GatedMILAttention
            pos_embeddings = []
            aggregators = []
            for dim in self.tile_encoder_dim:
                pos_embeddings.append(PositionalEmbedding(dim))
                aggregators.append(
                    nn.Sequential(
                        ABMILWrapper(abmil(dim, abmil_heads)),
                        nn.Flatten(1, -1),
                        nn.Linear(dim * abmil_heads, dim),
                    )
                )
            self.pos_embeddings = nn.ModuleList(pos_embeddings)
            self.aggregators = nn.ModuleList(aggregators)

        # get slide encoder dims if necessary
        if slide_encoder_dim is not None:
            self.slide_encoder_dim = (
                slide_encoder_dim
                if isinstance(slide_encoder_dim, list)
                else [slide_encoder_dim]
            )
            self.total_dim += sum(self.slide_encoder_dim)

        self.mlp = MLP(
            in_features=self.total_dim,
            hidden_dims=[1024, 512, 256],
            out_features=out_features,
        )

    def forward(
        self,
        tile_embeds: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        coords: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        slide_embeds: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    ) -> torch.Tensor:
        """
        Performs a forward pass.

        Parameters
        ----------
        tile_embeds : Union[torch.Tensor, List[torch.Tensor]]
            Input tile embeddings, shape (B, S, d_embed), where d_embed is
            input model dependent. If a list, the order of the embeddings
            must match the order of the tile_encoder_dim parameter from model
            init

        coords : Union[torch.Tensor, List[torch.Tensor]]
            Coordinates matching to input tile embed tensors, shape (B, S, 2)

        slide_embeds : Union[torch.Tensor, List[torch.Tensor]]
            Input slide embeddings, shape (B, d_embed), where d_embed is
            input model dependent. If a list, the order of the embeddings
            must match the order of the slide_encoder_dim parameter from model
            init

        Returns
        -------
        torch.Tensor
            The output, shape (B, d_out)
        """
        # put the slide embeds into a list if necessary
        if slide_embeds is not None:
            slide_embeds = (
                slide_embeds
                if isinstance(slide_embeds, list)
                else [slide_embeds]
            )
        else:
            slide_embeds = []

        # run the tile embeds through the aggregators
        if tile_embeds is not None:
            tile_embeds = (
                tile_embeds if isinstance(tile_embeds, list) else [tile_embeds]
            )
            coords = coords if isinstance(coords, list) else [coords]

            for i, embed in enumerate(tile_embeds):
                x = embed + self.pos_embeddings[i](
                    coords[i]
                )  # (B, S, d_embed)
                x = self.aggregators[i](x)  # (B, d_embed)
                slide_embeds.append(x)

        # concatenate the slide embeds and aggregator outputs and run through
        # MLP
        x = torch.cat(slide_embeds, dim=-1)  # (B, d_total)
        x = self.mlp(x)  # (B, d_out)
        return x
