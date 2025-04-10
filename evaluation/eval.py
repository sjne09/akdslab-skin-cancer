from enum import IntEnum
from math import ceil
from operator import itemgetter
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, auc

TYPE_CONFIG = {
    "ROC": {
        "curve_display": RocCurveDisplay,
        "set_initial_interp_value": True,
        "interp_transform": lambda x: x,
        "x_attr": "fpr",
        "y_attr": "tpr",
        "auc_attr": "roc_auc",
        "set_final_mean_value": True,
        "xlabel": "False Positive Rate",
        "ylabel": "True Positive Rate",
        "title": "Mean ROC curve with variability for {class_name}",
    },
    "PRC": {
        "curve_display": PrecisionRecallDisplay,
        "set_initial_interp_value": False,
        "interp_transform": lambda x: np.flip(x),
        "x_attr": "recall",
        "y_attr": "precision",
        "auc_attr": "average_precision",
        "set_final_mean_value": False,
        "xlabel": "Recall",
        "ylabel": "Precision",
        "title": (
            "Mean precision-recall curve with variability for {class_name}"
        ),
    },
}


class Evaluator:
    """
    A class for evaluating cross validated classifiers.
    """

    def __init__(
        self,
        labels: IntEnum,
        foundation_model: str,
        aggregator: str,
        classifier: str,
        onehot_labels: Dict[str, np.ndarray],
    ) -> None:
        """
        Initializes an Evaluator by creating dictionaries to store
        intermediate results and axes to plot results on.

        Parameters
        ----------
        labels : IntEnum
            The labels used for classification

        foundation_model : str
            The foundation model used in the experiment

        aggregator : str
            The aggregator used in the experiment

        classifier : str
            The classifier used in the experiment

        onehot_labels : Dict[str, np.ndarray]
            The onehot labels for each specimen, where the keys are
            specimen ids and values are the onehot labels
        """
        self.labels = labels
        self.foundation_model = foundation_model
        self.aggregator = aggregator
        self.classifier = classifier
        self.onehot_labels = onehot_labels

        # dictionaries to keep eval curve results for each label
        self.tprs = {label: [] for label in labels._member_names_}
        self.aucs = {label: [] for label in labels._member_names_}
        self.precisions = {label: [] for label in labels._member_names_}
        self.aps = {label: [] for label in labels._member_names_}

        # mean x vals and axes for curves
        self.mean_fpr = np.linspace(0, 1, 100)
        self.mean_recall = np.linspace(0, 1, 100)

        r = ceil(len(labels) / 2)
        self.roc_fig, self.roc_axs = plt.subplots(r, 2, figsize=(6 * r, 12))
        self.prc_fig, self.prc_axs = plt.subplots(r, 2, figsize=(6 * r, 12))

        # create empty results dataframe
        self.auroc_keys = [k + "_auroc" for k in labels._member_names_]
        self.auprc_keys = [k + "_auprc" for k in labels._member_names_]
        self.results = pd.DataFrame(
            columns=["foundation_model", "aggregator", "classifier", "fold"]
            + self.auroc_keys
            + self.auprc_keys
        )

    def fold(
        self,
        fold_idx: int,
        model_data: Dict[str, Any],
        num_folds: int,
    ) -> None:
        """
        Plot results from a single fold on the relevant axes and store results
        in class attribute dicts. Also updates the results dataframe.

        Parameters
        ----------
        fold_idx : int
            The index of the current fold being evaluated (0-indexed)

        model_data : Dict[str, Any]
            The best model data from the current fold. Includes the ids
            of the slides in the validation set (key 'ids'), the model
            outputs (key 'probs' - shape (N, C)) and the ground-truth
            labels (key 'labels')

        num_folds : int
            The total number of folds being evaluated
        """
        ids, probs = self.get_spec_level_probs(
            model_data["ids"], model_data["probs"]
        )
        labels_onehot = np.array(itemgetter(*ids)(self.onehot_labels))

        # for each predicted class, plot the current classifier's eval
        # and retain relevant data in dicts
        for j, class_of_interest in enumerate(self.labels._member_names_):
            interp_tpr, roc_auc = Evaluator.plot_eval(
                mean_x=self.mean_fpr,
                onehot_labels=labels_onehot[:, j],
                probs=probs[:, j],
                ax=self.roc_axs[j // 2][j % 2],
                plot_type="ROC",
                fold_idx=fold_idx,
                plot_chance_level=fold_idx == num_folds - 1,
            )
            self.tprs[class_of_interest].append(interp_tpr)
            self.aucs[class_of_interest].append(roc_auc)

            interp_precision, average_precision = Evaluator.plot_eval(
                mean_x=self.mean_recall,
                onehot_labels=labels_onehot[:, j],
                probs=probs[:, j],
                ax=self.prc_axs[j // 2][j % 2],
                plot_type="PRC",
                fold_idx=fold_idx,
                plot_chance_level=False,
            )
            self.precisions[class_of_interest].append(interp_precision)
            self.aps[class_of_interest].append(average_precision)

        # update results dataframe
        self._update_results(fold_idx)

    def _update_results(self, fold_idx: int) -> None:
        """
        Appends the results of the current fold to the results dataframe.

        Parameters
        ----------
        fold_idx : int
            The index of the current fold being evaluated (0-indexed)
        """
        # build results dict
        auroc_dict = {
            self.auroc_keys[i]: self.aucs[label][-1]
            for i, label in enumerate(self.labels._member_names_)
        }
        auprc_dict = {
            self.auprc_keys[i]: self.aps[label][-1]
            for i, label in enumerate(self.labels._member_names_)
        }
        model_details = {
            "foundation_model": self.foundation_model,
            "aggregator": self.aggregator,
            "classifier": self.classifier,
            "fold": fold_idx,
        }
        model_details = model_details | auroc_dict | auprc_dict

        # combine results with previous results
        details_df = pd.Series(model_details)
        self.results = pd.concat(
            [self.results, details_df.to_frame().T], ignore_index=True
        )

    def finalize(self, class_frequencies: Dict[str, float]) -> None:
        """
        Finalize the plots for each class by plotting the mean curves.

        Parameters
        ----------
        class_frequencies : Dict[str, float]
            Frequencies for each class, where keys are class names matching
            the member names of self.labels and values are the normalized
            frequencies
        """
        for j, class_of_interest in enumerate(self.labels._member_names_):
            Evaluator.create_mean_curve(
                self.mean_fpr,
                self.tprs[class_of_interest],
                self.aucs[class_of_interest],
                self.roc_axs[j // 2][j % 2],
                "ROC",
                class_of_interest,
            )

            prc_ax = self.prc_axs[j // 2][j % 2]
            Evaluator.create_mean_curve(
                self.mean_recall,
                self.precisions[class_of_interest],
                self.aps[class_of_interest],
                prc_ax,
                "PRC",
                class_of_interest,
            )

            # add chance line for PRC == label freq at specimen level
            prc_ax.axhline(
                class_frequencies[class_of_interest],
                linestyle="--",
                label=r"Chance level (AP = %0.2f)"
                % (class_frequencies[class_of_interest]),
                color="black",
            )

            # reorder legend
            handles, labs = prc_ax.get_legend_handles_labels()
            handles[-1], handles[-3] = handles[-3], handles[-1]
            labs[-1], labs[-3] = labs[-3], labs[-1]
            prc_ax.legend(handles=handles, labels=labs, loc="lower right")

    def save_figs(self, experiment_name: str) -> None:
        """
        Save the ROC and PRC plots to the outputs directory.

        Parameters
        ----------
        experiment_name : str
            The name of the experiment, used as an identifier for file name
        """
        self.roc_fig.savefig(f"outputs/{experiment_name}-roc.png")
        self.prc_fig.savefig(f"outputs/{experiment_name}-prc.png")

    @staticmethod
    def plot_eval(
        mean_x: np.ndarray,
        onehot_labels: np.ndarray,
        probs: np.ndarray,
        ax: plt.Axes,
        plot_type: str,
        fold_idx: int,
        plot_chance_level: bool,
    ) -> Tuple[np.ndarray, float]:
        """
        Plots the evaluation curve - either ROC or PRC - and returns the
        interpolated y-values based on the provided mean_x along with the area
        under the plotted curve.

        Parameters
        ----------
        mean_x : np.ndarray
            1D array with x values to be used to create a mean evaluation
            curve. Used along with provided probs to interpolate y values
            to enable averaging across multiple curves for different folds

        onehot_labels : np.ndarray
            Onehot labels corresponding to classes predicted with probs. Shape
            (N, C), where C is the number of classes

        probs : np.ndarray
            Classifier output probs. Shape (N, C) where C is the number of
            classes

        ax : plt.Axes
            The axis object to plot the curve on

        plot_type : str
            Specifies the type of evaluation curve to plot. Either "ROC" or
            "PRC"

        fold_idx : int
            The index of the current fold, used for the plot legend

        plot_chance_level : bool
            Whether to plot the chance level for the curve

        Returns
        -------
        np.ndarray
            1D array with interpolated y-values based on mean_x, to be used
            for creating a mean eval curve

        float
            The area under the curve
        """
        plot_type = plot_type.upper()
        config = TYPE_CONFIG.get(plot_type)
        if config is None:
            raise ValueError("type must be either 'ROC' or 'PRC'")

        # create the curve
        curve = config["curve_display"].from_predictions(
            onehot_labels,
            probs,
            name=f"{plot_type} fold {fold_idx}",
            plot_chance_level=plot_chance_level,
            ax=ax,
            alpha=0.3,
            lw=1,
        )

        # interpolate results using mean_x param to allow for averaging of
        # curves
        interp_y = np.interp(
            mean_x,
            config["interp_transform"](getattr(curve, config["x_attr"])),
            config["interp_transform"](getattr(curve, config["y_attr"])),
        )
        interp_y[0] = (
            0.0 if config["set_initial_interp_value"] else interp_y[0]
        )

        return interp_y, getattr(curve, config["auc_attr"])

    @staticmethod
    def create_mean_curve(
        mean_x: np.ndarray,
        cv_y: np.ndarray,
        aucs: np.ndarray,
        ax: plt.Axes,
        plot_type: str,
        class_name: str,
    ) -> None:
        """
        Creates a mean ROC or PRC curve based on the provided interpolated
        y-values from cross validation folds. Also plots standard deviation
        bands around the mean curve.

        Parameters
        ----------
        mean_x : np.ndarray
            1D array with x values to be used to create a mean evaluation
            curve

        cv_y : np.ndarray
            Interpolated y-values for each cross validation fold, shape
            (F, N), where F is the number of folds and N is the number of
            examples in the validation set

        aucs : np.ndarray
            AUCs for each fold, shape (F,)

        ax : plt.Axes
            The axis object to plot the curve on

        plot_type : str
            Specifies the type of evaluation curve to plot. Either "ROC" or
            "PRC"

        class_name : str
            The class name to be used in the chart title
        """
        plot_type = plot_type.upper()
        config = TYPE_CONFIG.get(plot_type)
        if config is None:
            raise ValueError("type must be either 'ROC' or 'PRC'")

        ax.set(
            xlabel=config["xlabel"],
            ylabel=config["ylabel"],
            title=config["title"].format(class_name=class_name),
        )

        # prep plot data
        mean_y = np.mean(cv_y, axis=0)
        mean_y[-1] = 1.0 if config["set_final_mean_value"] else mean_y[-1]
        mean_auc = auc(mean_x, mean_y)
        std_auc = np.std(aucs)
        auc_label = "AUC" if plot_type == "ROC" else "AP"

        # plot the mean line
        ax.plot(
            mean_x,
            mean_y,
            color="b",
            label=r"Mean %s (%s = %0.2f $\pm$ %0.2f)"
            % (plot_type, auc_label, mean_auc, std_auc),
            lw=2,
            alpha=0.8,
        )

        # plot the standard deviation bands
        std_y = np.std(cv_y, axis=0)
        y_upper = np.minimum(mean_y + std_y, 1)
        y_lower = np.maximum(mean_y - std_y, 0)
        ax.fill_between(
            mean_x,
            y_lower,
            y_upper,
            color="grey",
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
        )

        ax.legend(loc="lower right")

    @staticmethod
    def get_spec_level_probs(
        slide_indices: List[str], probs: np.ndarray
    ) -> Tuple[List[str], np.ndarray]:
        """
        Calculates the average model output probabilities for each class
        at the specimen level rather than slide level.

        Parameters
        ----------
        slide_indices : List[str]
            A list of slide ids for which the specimen is the first 6 chars

        probs : np.ndarray
            Classifier output probabilities for each slide included in
            slide_indices; shape (N, O) where N is the number of slides
            and O is the number of classes

        Returns
        -------
        List[str]
            The specimen ids

        np.ndarray
            The average probabilities for each specimen; shape (M, O) where
            M is the number of specimens and O is the number of classes
        """
        combined_probs = {}
        for j, slide_idx in enumerate(slide_indices):
            spec = slide_idx[:6]
            if combined_probs.get(spec) is None:
                combined_probs[spec] = [probs[j]]
            else:
                combined_probs[spec].append(probs[j])

        for k in combined_probs.keys():
            combined_probs[k] = np.array(combined_probs[k])
            combined_probs[k] = (
                combined_probs[k].sum(axis=0) / combined_probs[k].shape[0]
            )

        return list(combined_probs.keys()), np.array(
            list(combined_probs.values())
        )
