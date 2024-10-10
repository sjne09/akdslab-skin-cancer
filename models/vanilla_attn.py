import torch
from torch import nn

from .MLP import MLP
from .pos_embedding import PositionalEmbedding


class VanillaAttentionAggregator(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        out_features: int,
        num_heads: int,
        global_pooling: bool = False,
    ) -> None:
        """
        Uses CLS token mechanism from ViT.

        Parameters
        ----------
        embed_dim : int
            The number of features in the input

        out_features : int
            The number of features in the output

        num_heads : int
            The number of attention heads
        """
        super().__init__()
        self.global_pooling = global_pooling
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos = PositionalEmbedding(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.mlp = MLP(
            in_features=embed_dim,
            hidden_dims=[embed_dim, embed_dim // 2, embed_dim // 4],
            out_features=out_features,
        )

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input data, shape (B, S, d_embed)

        coords : torch.Tensor
            Coordinates matching to input tensor, shape (B, S, 2)

        Returns
        -------
        torch.Tensor
            The output, shape (B, d_out)
        """
        B, S, D = x.shape

        # add position embeddings prior to concating CLS token
        x = x + self.pos(coords)  # (B, S, d_embed)

        # concat CLS tokens with x
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, S+1, d_embed)

        att, _ = self.attn(x, x, x)  # (B, S+1, d_embed)

        if self.global_pooling:
            x = att[:, 1:, :].mean(dim=1)  # (B, d_embed)
        else:
            x = att[:, 0, :]  # (B, d_embed)

        x = self.mlp(x)
        return x
