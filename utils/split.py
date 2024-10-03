from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def train_val_split(
    X: pd.DataFrame,
    y: pd.DataFrame,
    val_fold_indices: List[str],
    z_norm: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct training and validation sets from the provided data using
    the given indices for the validation set.

    Parameters
    ----------
    X : pd.DataFrame
        The feature set

    y : pd.DataFrame
        The targets

    val_fold_indices : List[str]
        The indices of samples to include in the validation set

    z_norm : bool
        Whether to use z-score normalization / standardization

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        X_train, y_train, X_val, y_val
    """
    X_split = {}
    y_split = {}
    X_split["val"] = X.loc[val_fold_indices]
    y_split["val"] = y.loc[val_fold_indices]
    X_split["train"] = X[X.index.difference(X_split["val"].index)]
    y_split["train"] = y[y.index.difference(y_split["val"].index)]
    assert X_split["val"].index == y_split["val"].index
    assert X_split["train"].index == y_split["train"].index

    for split, embs in X_split.items():
        embs = np.stack(embs.to_list())
        if z_norm:
            embs = (embs - embs.mean(axis=1, keepdims=True)) / embs.std(
                axis=1, keepdims=True
            )

        X_split[split] = embs
        y_split[split] = y_split[split].to_numpy()

    return (
        X_split["train"],
        y_split["train"],
        X_split["val"],
        y_split["val"],
    )


def train_val_split_slides(
    val_fold: int, spec_folds: List[List[str]], specs: Dict[str, List[str]]
) -> Tuple[List[str], List[str]]:
    val = [slide for spec in spec_folds[val_fold] for slide in specs[spec]]
    train = [
        slide
        for i, fold in enumerate(spec_folds)
        for spec in fold
        for slide in specs[spec]
        if i != val_fold
    ]
    return train, val


def train_val_split_labels(
    val_fold: int, data: pd.DataFrame, spec_folds: List[List[str]]
) -> Tuple[Dict[str, int], Dict[str, int]]:
    val_labels = {
        spec: int(data.loc[spec]["label"]) for spec in spec_folds[val_fold]
    }
    train_labels = {
        spec: int(data.loc[spec]["label"])
        for i, fold in enumerate(spec_folds)
        for spec in fold
        if i != val_fold
    }
    return train_labels, val_labels
