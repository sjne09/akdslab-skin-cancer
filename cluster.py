import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from umap import UMAP

from utils.Label import Label
from utils.load_data import load_data

load_dotenv()

OUTPUT_DIR = os.getenv("OUTPUT_DIR")
DATA_DIR = os.getenv("DATA_DIR")


def cluster_embeds(
    embedding_path: str,
    label_path: str,
    model_name: str,
    save_path: str,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> None:
    """
    Saves a UMAP projection of embeddings to disk.

    Parameters
    ----------
    embedding_path : str
        Path to pickled slide embeddings. Pickled file must contain a dict
        with slide ids as keys and embeddings as values

    label_path : str
        Path to the label data csv

    model_name : str
        The name of the model used to generate the slide embeddings

    save_path : str
        Location to save the output projection to

    n_neighbors : int
        The size of the neighborhood for UMAP. Lower values = local structure,
        higher values = global stucture

    min_dist : float
        Minimum distance between embedded points for UMAP. Lower values =
        tighter clusers, higher values = more even dispersal
    """
    df = load_data(embedding_path=embedding_path, label_path=label_path)
    reducer = UMAP(n_neighbors=n_neighbors, min_dist=min_dist)

    emb = reducer.fit_transform(df["embedding"].to_list())
    plt.scatter(
        emb[:, 0],
        emb[:, 1],
        c=[sns.color_palette()[x] for x in df["label"]],
        s=5,
    )

    # create the legend based on labels
    patches = []
    for i, label in enumerate(Label._member_names_):
        patches.append(
            mpatches.Patch(color=sns.color_palette()[i], label=label)
        )

    plt.gca().set_aspect("equal", "datalim")
    plt.title(f"UMAP projection of {model_name} slide embeds")
    plt.legend(handles=patches)
    plt.figtext(
        0.99,
        0.01,
        f"n_neighbors = {reducer.n_neighbors}, min_dist = {reducer.min_dist}",
        ha="right",
        color="gray",
        alpha=0.5,
    )
    plt.savefig(save_path, dpi=200)


if __name__ == "__main__":
    label_path = os.path.join(DATA_DIR, "labels/labels.csv")
    embedding_path = os.path.join(OUTPUT_DIR, "gigapath_slide_embeds_cls.pkl")
    cluster_embeds(
        embedding_path,
        label_path,
        "gigapath",
        "outputs/gigapath/cls/gigapath_proj.png",
        50,
    )
