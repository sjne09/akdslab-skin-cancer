from functools import partial
from typing import Callable, Dict, List, Tuple, Type, Union

from torch.utils.data import DataLoader, Dataset

from data_processing.split import (
    train_val_split_labels,
    train_val_split_slides,
)


def get_loaders(
    val_fold: int,
    specimens_by_fold: List[List[str]],
    slides_by_specimen: Dict[str, List[str]],
    labels_by_specimen: Dict[str, int],
    train_dataset_class: Union[Type[Dataset], partial],
    val_dataset_class: Union[Type[Dataset], partial],
    collate_fn: Callable[[list], dict],
) -> Tuple[DataLoader, DataLoader]:
    """
    Returns training and validation dataloaders for a given fold.

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

    train_dataset_class : Union[Type[Dataset], partial]
        The class to use for the training dataset. Can be a partial
        function with arguments

    val_dataset_class : Union[Type[Dataset], partial]
        The class to use for the validation dataset. Can be a partial
        function with arguments

    collate_fn : Callable[[list], dict]
        The function to use for collating the data

    Returns
    -------
    Tuple[DataLoader, DataLoader]
        The training and validation dataloaders
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

    train_loader = DataLoader(
        train_dataset_class(slide_ids=train, labels=train_labels),
        batch_size=1,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset_class(slide_ids=val, labels=val_labels),
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader
