import os
import pickle
from typing import Dict, Iterator, List, Tuple

import numpy as np
import timm
import torch
from dotenv import load_dotenv
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

load_dotenv()
DATA_DIR = os.getenv("DATA_DIR")
OUTPUT_DIR = os.getenv("OUTPUT_DIR")


class TileEncodingDataset(Dataset):
    """
    Taken from prov-gigapath pipeline
    """

    def __init__(
        self, image_paths: List[str], transform: transforms.Compose = None
    ) -> None:
        self.transform = transform
        self.image_paths = image_paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path = self.image_paths[idx]
        img_name = os.path.basename(img_path)
        # get x, y coordinates from the image name
        x, y = img_name.split(".png")[0].split("_")
        x, y = int(x.replace("x", "")), int(y.replace("y", ""))
        # load the image
        with open(img_path, "rb") as f:
            img = Image.open(f).convert("RGB")
            if self.transform:
                img = self.transform(img)
        return {
            "img": torch.from_numpy(np.array(img)),
            "coords": torch.from_numpy(np.array([x, y])).float(),
        }


def run_inference(
    tile_paths: List[str],
    model: nn.Module,
    transform: transforms.Compose,
    batch_size: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Runs inference using a tile embedding model.

    Parameters
    ----------
    tile_paths : List[str]
        A list of paths to tiles of a single WSI

    model : nn.Module
        The tile embedding model

    transform : transforms.Compose
        Transforms to apply to the tiles

    batch_size : int
        Batch size to use

    device : torch.device
        The device to send the model and images to

    Returns
    -------
    Dict[str, torch.Tensor]
        A dict with keys "tile_embeds" and "coords" and tensor values with
        shapes (N, D) and (N, 2) where N is the number of tiles for the slide
        and D is the output dimension from the model
    """
    model.to(device)

    loader = DataLoader(
        TileEncodingDataset(tile_paths, transform=transform),
        batch_size=batch_size,
        shuffle=False,
    )

    model.eval()
    collated_outputs = {"tile_embeds": [], "coords": []}
    with torch.inference_mode():
        for batch in loader:
            collated_outputs["tile_embeds"].append(
                model(batch["img"].to(device)).detach().cpu()
            )
            collated_outputs["coords"].append(batch["coords"])

    return {k: torch.cat(v) for k, v in collated_outputs.items()}


def create_tile_embeds(
    tile_encoder_model: nn.Module,
    transform: transforms.Compose,
    tiles_dir: str,
    num_gpus: int,
    device: torch.device,
) -> None:
    """
    Create tile embeddings for each slide in the provided tiles directory and
    save to disk.

    Parameters
    ----------
    tile_encoder_model : nn.Module
        The tile encoder

    tiles_dir : str
        The directory containing subdirectories with tiles. Each subdirectory
        corresponds to a single slide

    num_gpus : int
        The number of GPUs to run on
    """
    for dirname in os.listdir(tiles_dir):
        dpath = os.path.join(tiles_dir, dirname)
        if os.path.isdir(dpath):
            tiles = [
                os.path.join(dpath, fname)
                for fname in os.listdir(dpath)
                if fname.endswith(".png")
            ]
            if len(tiles) > 0:
                print(f"running for {dpath} with {len(tiles)} tiles")

                # inference method will move model and tiles to cuda device;
                # model return includes tensor containing embeddings and
                # tensor containing tile coords
                embeds = (
                    dirname,
                    run_inference(
                        tiles,
                        tile_encoder_model,
                        transform,
                        128 * num_gpus,
                        device,
                    ),
                )

                # pickle the embeddings
                with open(
                    os.path.join(
                        OUTPUT_DIR,
                        "uni/tile_embeddings/{name}.pkl".format(
                            name=os.path.splitext(dirname)[0]
                        ),
                    ),
                    "wb",
                ) as f:
                    pickle.dump(embeds[1], f)


def load_model(device: torch.device) -> Tuple[nn.Module, transforms.Compose]:
    """
    Loads the UNI model and transforms required for inference.

    Parameters
    ----------
    device : torch.device
        The map location for the model

    Returns
    -------
    Tuple[nn.Module, transforms.Compose]
        The UNI tile encoder model and transforms required for inference
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
            map_location=device,
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


def load_tile_embeds(
    tile_embeds_path: str,
) -> Iterator[Tuple[str, Dict[str, torch.Tensor]]]:
    """
    Yields tile embeddings for a single slide at a time by accessing pickled
    tile embeddings.

    Parameters
    ----------
    tile_embeds_path : str
        The path to the pickled tile embeddings. Each pickle file should
        contain the collated tile embeddings for a single slide in a dict
        with keys "tile_embeds" and "coords" and tensor values with shapes
        (N, D) and (N, 2) where N is the number of tiles for the slide and D
        is the output dimension from the UNI model

    Yields
    ------
    Tuple[str, Dict[str, torch.Tensor]]
        A tuple containing the slide id and the embedding dict with keys
        "tile_embeds" and "coords"
    """
    for slide in os.listdir(tile_embeds_path):
        name = os.path.splitext(slide)[0]
        with open(os.path.join(tile_embeds_path, slide), "rb") as f:
            # each .pkl file contains a dict with keys "tile_embeds",
            # "coords" and values torch.Tensor
            emb = pickle.load(f)
            yield (name, emb)


def create_pooled_slide_embeds(
    tile_embeds_path: str, z_norm: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Creates slide embeddings using the global pooling strategy (i.e., by
    averaging all of the tile embeddings).

    Parameters
    ----------
    tile_embeds_path : str
        The path to the pickled tile embeddings. Each pickle file should
        contain the collated tile embeddings for a single slide in a dict
        with keys "tile_embeds" and "coords" and tensor values with shapes
        (N, D) and (N, 2) where N is the number of tiles for the slide and D
        is the output dimension from the UNI model

    Returns
    -------
    Dict[str, torch.Tensor]
        A dict with slide ids as keys and pooled embeddings as values
    """
    slide_embeds = {}
    for name, emb in load_tile_embeds(tile_embeds_path):
        slide_embeds[name] = emb["tile_embeds"].mean(dim=0)
        if z_norm:
            slide_embeds[emb] = (
                slide_embeds[emb] - slide_embeds[emb].mean()
            ) / slide_embeds[emb].std()
        return slide_embeds


def main():
    gpus = ["1", "2"]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_processes = len(gpus)
    print(f"Using device {device}")

    # load model and run it
    model, transform = load_model(device)
    model = nn.DataParallel(model)
    print("---")

    create_tile_embeds(
        model,
        transform,
        os.path.join(DATA_DIR, "tiles/output"),
        num_processes,
        device,
    )


if __name__ == "__main__":
    main()
