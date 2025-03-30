from typing import Dict, List, Tuple, Union

import numpy as np

from data_processing.data_utils import load_pickled_embeds


def train_val_split_sk_clf(
    val_fold: int,
    specimens_by_fold: List[List[str]],
    slides_by_specimen: Dict[str, List[str]],
    labels_by_specimen: Dict[str, int],
    embedding_path: str,
) -> Dict[str, Union[np.ndarray, List[str]]]:
    """
    Split data into training and validation sets for a scikit-learn
    classifier. This function is specifically for slide embeddings.

    Parameters
    ----------
    val_fold : int
        The fold to use for validation

    specimens_by_fold : List[List[str]]
        The list of specimens in each fold

    slides_by_specimen : Dict[str, List[str]]
        The slides associated with each specimen, with the specimen ids
        as keys and the slide ids as values

    labels_by_specimen : Dict[str, int]
        The labels associated with each specimen, with the specimen ids
        as keys and the labels as values

    embedding_path : str
        The path to the pickle file containing the slide embeddings

    Returns
    -------
    Dict[str, Union[np.ndarray, List[str]]]
        The training and validation sets of embeddings, labels, and
        slide ids
    """
    train, val = train_val_split_slides(
        val_fold=val_fold,
        specimens_by_fold=specimens_by_fold,
        slides_by_specimen=slides_by_specimen,
    )

    train_labels, val_labels = train_val_split_labels(
        val_fold=val_fold,
        labels_by_specimen=labels_by_specimen,
        specimens_by_fold=specimens_by_fold,
    )

    slide_embeds = load_pickled_embeds(embedding_path)

    X_train = np.stack([slide_embeds[slide] for slide in train])
    X_val = np.stack([slide_embeds[slide] for slide in val])
    y_train = np.array([train_labels[slide[:6]] for slide in train])
    y_val = np.array([val_labels[slide[:6]] for slide in val])

    fold_data = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "train_ids": train,
        "val_ids": val,
    }

    return fold_data


def train_val_split_slides(
    val_fold: int,
    specimens_by_fold: List[List[str]],
    slides_by_specimen: Dict[str, List[str]],
) -> Tuple[List[str], List[str]]:
    """
    Split slides into training and validation sets based on the
    provided folds. Splits are at the slide level.

    Parameters
    ----------
    val_fold : int
        The fold to use for validation

    specimens_by_fold : List[List[str]]
        The list of specimens in each fold

    slides_by_specimen : Dict[str, List[str]]
        The slides associated with each specimen, with the specimen ids
        as keys and the slide ids as values

    Returns
    -------
    Tuple[List[str], List[str]]
        The training and validation sets of slides
    """
    val = [
        slide
        for spec in specimens_by_fold[val_fold]
        for slide in slides_by_specimen[spec]
    ]
    train = [
        slide
        for i, fold in enumerate(specimens_by_fold)
        for spec in fold
        for slide in slides_by_specimen[spec]
        if i != val_fold
    ]
    return train, val


def train_val_split_labels(
    val_fold: int,
    labels_by_specimen: Dict[str, int],
    specimens_by_fold: List[List[str]],
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Split labels into training and validation sets based on the
    provided folds. Labels are at the specimen level

    Parameters
    ----------
    val_fold : int
        The fold to use for validation

    labels_by_specimen : Dict[str, int]
        The labels associated with each specimen, with the specimen ids
        as keys and the labels as values

    specimens_by_fold : List[List[str]]
        The list of specimens in each fold

    Returns
    -------
    Tuple[Dict[str, int], Dict[str, int]]
        The training and validation sets of labels, with slide ids as
        keys and labels as values
    """
    val_labels = {
        spec: labels_by_specimen[spec] for spec in specimens_by_fold[val_fold]
    }
    train_labels = {
        spec: labels_by_specimen[spec]
        for i, fold in enumerate(specimens_by_fold)
        for spec in fold
        if i != val_fold
    }
    return train_labels, val_labels
