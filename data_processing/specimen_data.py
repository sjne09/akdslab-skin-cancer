from random import randint
from typing import Dict, List, Optional

import numpy as np

from data_processing.data_utils import load_data
from data_processing.label import Label


class SpecimenData:
    def __init__(
        self, label_path: str, fold_path: Optional[str] = None
    ) -> None:
        """
        Parameters
        ----------
        label_path : str
            Path to the label data csv

        fold_path : Optional[str]
            Path to fold mapping json. Json file must contain a dict
            with fold indices as keys and a list of specimen ids as
            values
        """
        df = load_data(label_path=label_path, fold_path=fold_path)
        self.df = df.set_index("specimen_id")
        self.specimens = list(self.df.index)
        self.onehot_labels = self._get_onehot_labels()
        self.labels = self._get_labels()
        self.specimens_by_label = self._get_specimens_by_label()
        self.class_freqs = self._get_class_freqs()
        if fold_path is not None:
            self.specimens_by_fold = self._get_specimens_by_fold()
            self.n_folds = len(self.specimens_by_fold)

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

    def _get_specimens_by_label(self) -> Dict[int, List[str]]:
        """
        Returns a dict that groups specimens by label.

        Returns
        -------
        Dict[int, List[str]]
            A mapping of labels to specimen ids
        """
        specimens_by_label = self.df.groupby("label").groups
        specimens_by_label = [
            list(specs) for specs in specimens_by_label.values()
        ]
        return

    def _get_class_freqs(self) -> Dict[int, float]:
        """
        Returns the frequency of each class in the dataset.

        Returns
        -------
        Dict[int, float]
            The frequency of each class in the dataset
        """
        return {
            label: self.df[label].value_counts(normalize=True).iloc[1]
            for label in Label._member_names_
        }

    def sample_specs(self, n: int) -> Dict[int, List[str]]:
        """
        Returns a random sample of n specimens for each unique label, drawn
        without replacement.

        Parameters
        ----------
        n : int
            The number of samples to draw per label

        Returns
        -------
        Dict[int, List[str]]
            The random samples, structured as a dict with label ids as keys
            and a list of specimen ids as values
        """
        sample = {i: [] for i in range(len(self.specimens_by_label))}
        for i in range(len(self.specimens_by_label)):
            for _ in range(n):
                idx = randint(0, len(self.specimens_by_label[i]) - 1)
                sample[i].append(self.specimens_by_label[i][idx])
        return sample
