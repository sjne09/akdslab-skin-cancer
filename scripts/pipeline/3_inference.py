import gc
import os
from functools import partial
from typing import Type, Union

import torch

from models.foundation_models import (
    PRISM,
    UNI,
    FoundationModel,
    GigaPath,
    ResNet18,
)

DATA_DIR = os.environ["DATA_DIR"]
OUTPUT_DIR = os.environ["OUTPUT_DIR"]


def initialize_fm(
    fm: Union[Type[FoundationModel], partial], name: str
) -> FoundationModel:
    """
    Initializes a FoundationModel with the appropriate paths.

    Parameters
    ----------
    fm : Union[Type[FoundationModel], partial]
        The FoundationModel class to initialize

    name : str
        The name of the FoundationModel

    Returns
    -------
    FoundationModel
        The initialized FoundationModel
    """
    # create the output directories if necessary
    tile_embeds_path = os.path.join(
        OUTPUT_DIR, f"{name.lower()}/tile_embeddings"
    )
    slide_embeds_path = os.path.join(
        OUTPUT_DIR, f"{name.lower()}/slide_embeddings"
    )
    try:
        os.makedirs(tile_embeds_path)
        os.makedirs(slide_embeds_path)
    except FileExistsError:
        pass

    # instantiate the model
    return fm(
        tiles_dir=os.path.join(DATA_DIR, "tiles/output"),
        tile_embeds_path=tile_embeds_path,
        slide_embeds_path=slide_embeds_path,
    )


def main() -> None:
    """
    Runs inference on all FoundationModels.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    # loop variables
    slide_embed_names = {
        UNI: None,
        PRISM: "perceiver",
        GigaPath: "pool",
        partial(GigaPath, global_pool=False): "cls",
        ResNet18: None,
    }
    fms = slide_embed_names.keys()

    for fm in fms:
        name = fm.func.__name__ if isinstance(fm, partial) else fm.__name__
        print(f"Running {name}")
        model = initialize_fm(fm, name)

        # run tile inference
        model.create_tile_embeds(device)
        print("Finished tile embeddings")

        # run GAP slide inference
        model.create_pooled_slide_embeds(
            fname=f"{name.lower()}_slide_embeds_GAP"
        )
        print("Finished GAP slide embeddings")

        # run slide inference with fm-specific models if available
        try:
            model.create_slide_embeds(
                fname=(
                    f"{name.lower()}_slide_embeds_" f"{slide_embed_names[fm]}"
                ),
                device=device,
            )
            print(f"Finished {slide_embed_names[fm]} slide embeddings")
        except NotImplementedError:
            pass

        # clean up
        del model
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Finished {name}")
        print()

    print("Finished all models")


if __name__ == "__main__":
    main()
