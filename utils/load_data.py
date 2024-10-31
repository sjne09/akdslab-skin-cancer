import json
import os
import pickle
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from data_models.Label import Label

load_dotenv()

OUTPUT_DIR = os.getenv("OUTPUT_DIR")
DATA_DIR = os.getenv("DATA_DIR")


class SpecimenData:
    def __init__(self, label_path: str, fold_path: str):
        df = load_data(label_path=label_path, fold_path=fold_path)
        self.df = df.set_index("specimen_id")
        self.specimens = list(self.df.index)
        self.onehot_labels = self._get_onehot_labels()
        self.labels = self._get_labels()
        self.specimens_by_fold = self._get_specimens_by_fold()

    def _get_onehot_labels(self) -> Dict[str, np.ndarray]:
        """
        Returns a dictionary of specimen ids to one-hot encoded labels.

        Returns
        -------
        Dict[str, np.ndarray]
            A dictionary of specimen ids to one-hot encoded labels
        """
        onehot_labels = self.df[Label._member_names_].to_dict(
            orient="split", index=True
        )
        onehot_labels = {
            k: np.array(onehot_labels["data"][i])
            for i, k in enumerate(onehot_labels["index"])
        }
        return onehot_labels

    def _get_labels(self) -> Dict[str, int]:
        """
        Returns a dictionary of specimen ids to label indices.

        Returns
        -------
        Dict[str, int]
            A dictionary of specimen ids to label indices
        """
        labels = {row.name: int(row["label"]) for _, row in self.df.iterrows()}
        return labels

    def _get_specimens_by_fold(self) -> List[List[str]]:
        """
        Returns a list of lists of specimen ids, grouped by fold index.

        Returns
        -------
        List[List[str]]
            A list of lists of specimen ids, grouped by fold index
        """
        specimens_by_fold = self.df.groupby("fold").groups
        specimens_by_fold = [
            list(specs) for specs in specimens_by_fold.values()
        ]
        return specimens_by_fold


def get_label(x: pd.DataFrame) -> int:
    """
    Returns a target label index for a single dataframe entry.

    Parameters
    ----------
    x : pd.DataFrame
        A single row of a dataframe, with one-hot columns for "bowens",
        "bcc", and "scc"

    Returns
    -------
    int
        The label index
    """
    if x["bowens"] == 1:
        return Label.bowens.value
    elif x["bcc"] == 1:
        return Label.bcc.value
    elif x["scc"] == 1:
        return Label.scc.value
    else:
        return Label.na.value


def load_embedding_df(embedding_path: str) -> pd.DataFrame:
    """
    Loads a pandas dataframe with slide embeddings.

    Parameters
    ----------
    embedding_path : str
        Path to pickled slide embeddings. Pickled file must contain a dict
        with slide ids as keys and embeddings as values

    Returns
    -------
    pd.DataFrame
        The loaded data with columns for slide id, specimen id, and embedding
    """
    # load the pickled dict
    with open(os.path.join(embedding_path), "rb") as f:
        embeds = pickle.load(f)

    # extract data from dict for easy dataframe creation
    slide_ids = list(embeds.keys())
    specimen_ids = [slide_id[:6] for slide_id in slide_ids]
    embeddings = list(embeds.values())

    # create the dataframe
    df = pd.DataFrame(
        {
            "slide_id": slide_ids,
            "specimen_id": specimen_ids,
            "embedding": embeddings,
        }
    )
    df["slide_id"] = df["slide_id"].astype("string")
    df["specimen_id"] = df["specimen_id"].astype("string")
    df["embedding"] = df["embedding"].apply(lambda x: np.array(x.squeeze(0)))

    return df


def load_data(
    label_path: str,
    embedding_path: Optional[str] = None,
    fold_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Loads the data from label_path into a pandas dataframe. If embedding_path
    is provided, load embeddings as well and merge into dataframe. If
    fold_path is provided, load fold index mapping and merge into dataframe.

    Parameters
    ----------
    label_path : str
        Path to the label data csv

    embedding_path : Optional[str]
        Path to pickled slide embeddings. Pickled file must contain a dict
        with slide ids as keys and embeddings as values

    fold_path : Optional[str]
        Path to fold mapping json. Json file must contain a dict with fold
        indices as keys and a list of specimen ids as values

    Returns
    -------
    pd.DataFrame
        The loaded data
    """
    df = (
        pd.read_csv(label_path)
        .rename(
            columns={
                "patientid": "patient_id",
                "specimenid": "specimen_id",
                "bowens1": "bowens",
                "scc1": "scc",
                "bcc1": "bcc",
            }
        )
        .drop("box", axis=1)
    )

    # set column types to allow for easier merging, etc.
    df["specimen_id"] = df["specimen_id"].astype("string")
    df["patient_id"] = df["patient_id"].astype("string")

    # define column reflecting non-cancerous sample and drop superfluous data
    df["na"] = df["nmsc1"].map(lambda x: 1 - x)
    df = df.drop("nmsc1", axis=1)

    # get target labels for each row
    df["label"] = df.apply(get_label, axis=1)

    # add embeddings for each slide if embedding path provided
    if embedding_path is not None:
        embed_df = load_embedding_df(embedding_path)

        df = embed_df.merge(df, how="right", on="specimen_id")
        df = df.set_index("slide_id")

    # add fold indices for each specimen/slide if fold path provided
    if fold_path is not None:
        with open(fold_path, "r") as f:
            folds = json.load(f)
        fold_map = {spec: i for i, fold in enumerate(folds) for spec in fold}
        df["fold"] = df["specimen_id"].map(lambda x: fold_map[x])

    return df
