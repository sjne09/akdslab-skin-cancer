import os
from typing import Any, Dict, Iterable, List, Optional

from gigapath.preprocessing.data.foreground_segmentation import LoadROId
from matplotlib import collections, patches
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.image import AxesImage
from monai.data.wsi_reader import WSIReader


def slide_id_from_path(fpath: str) -> str:
    """
    Extracts the slide id from the slide image file path.

    Parameters
    ----------
    fpath : str
        The slide image file path

    Returns
    -------
    str
        The slide id
    """
    return os.path.splitext(fpath.split("/")[-1])[0]


def load_slide(fpath: str) -> Dict[str, Any]:
    """
    Loads a WSI using gigapath's preprocessing pipeline.

    Parameters
    ----------
    fpath : str
        The slide image file path

    Returns
    -------
    Dict[str, Any]
        A dict containing the following keys and data:
            "image" : np.ndarray
                The image
            "slide-id" : str
                The slide id
            "location" : Tuple[int, int]
                ???
            "size" : Tuple[int, int]
                The size of the image in pixels
            "level" : int
                The level that the image data corresponds to
            "origin" : Tuple[int, int]
                The origin for the loaded image; may differ from (0, 0) in the
                case that the image is cropped when processed
            "scale" : float
                The downscale factor of the image
            "foreground_threshold" : float
                The luminance threshold used in preprocessing to determine
                tile occupancy
    """
    sample = {
        "image": fpath,
        "slide_id": slide_id_from_path(fpath),
    }
    loader = LoadROId(
        WSIReader(backend="OpenSlide"),
        level=0,
        margin=0,
        foreground_threshold=None,
    )
    sample = loader(sample)
    return sample


def plot_image(
    fpath: str,
    ax: plt.Axes,
    tile_coords: Optional[List[Iterable[int]]] = None,
    tile_weights: Optional[List[float]] = None,
    weight_labels: Optional[Dict[str, int]] = None,
) -> AxesImage:
    sample = load_slide(fpath)

    slide_image = sample["image"]
    downscale_factor = sample["scale"]

    im = ax.imshow(slide_image.transpose(1, 2, 0))

    if tile_coords is not None:
        rects = []
        for x, y in tile_coords:
            # change coordinate to the current level from level-0
            # tile location is in the original image coordinate, while the
            # slide image is after selecting ROI
            xy = (
                (x - sample["origin"][0]) / downscale_factor,
                (y - sample["origin"][1]) / downscale_factor,
            )
            rects.append(patches.Rectangle(xy, 256, 256))

        if tile_weights is not None:
            pc = collections.PatchCollection(rects, alpha=1, cmap="inferno")
            pc.set_array(tile_weights)

            # Add a custom legend
            if weight_labels is not None:
                cmap = plt.cm.inferno
                norm = Normalize(
                    vmin=min(weight_labels.values()),
                    vmax=max(weight_labels.values()),
                )
                pc.set_clim(
                    vmin=min(weight_labels.values()),
                    vmax=max(weight_labels.values()),
                )
                legend_handles = [
                    patches.Patch(color=cmap(norm(value)), label=name)
                    for name, value in weight_labels.items()
                ]
                ax.legend(
                    handles=legend_handles,
                    title="Tile Classification",
                    loc="upper right",
                    bbox_to_anchor=(1.2, 0.6),
                )
            else:
                plt.colorbar(pc, ax)
        else:
            pc = collections.PatchCollection(
                rects, match_original=True, alpha=0.5, edgecolor="black"
            )
        ax.add_collection(pc)
    ax.axis("off")
    return im
