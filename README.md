**Note:** in timm.models.vision_transformer, must change `vit_giant_patch14_dinov2()` to the following:

```python
@register_model
def vit_giant_patch14_dinov2(pretrained=False, **kwargs) -> VisionTransformer:
    """ ViT-G/14 for DINOv2
    """

    # The hidden_features of SwiGLU is calculated by:
    # hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
    # When embed_dim=1536, hidden_features=4096
    # With SwiGLUPacked, we need to set hidden_features = 2 * 4096 = 8192
    cfg: dict = kwargs.get("pretrained_cfg", {})
    cfg["img_size"] = cfg["input_size"][1]
    model_args = dict(
        patch_size=cfg.pop("patch_size", 14), 
        embed_dim=cfg.pop("embed_dim", 1536), 
        depth=cfg.pop("depth", 40), 
        num_heads=cfg.pop("num_heads", 24), 
        init_values=cfg.pop("init_values", 1e-5),
        mlp_ratio=cfg.pop("mlp_ratio", 2.66667 * 2), 
        mlp_layer=cfg.pop("mlp_layer", SwiGLUPacked), 
        img_size=cfg.pop("img_size", 518), 
        act_layer=cfg.pop("act_layer", nn.SiLU),
    )
    model = _create_vision_transformer(
        'vit_giant_patch14_dinov2', pretrained=pretrained, **dict(model_args, **kwargs))
    return model
```
