from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
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
        1D array with x values to be used to create a mean evaluation curve.
        Used along with provided probs to interpolate y values to enable
        averaging across multiple curves for different folds

    onehot_labels : np.ndarray
        Onehot labels corresponding to classes predicted with probs. Shape
        (N, C), where C is the number of classes

    probs : np.ndarray
        Classifier output probs. Shape (N, C) where C is the number of classes

    ax : plt.Axes
        The axis object to plot the curve on

    plot_type : str
        Specifies the type of evaluation curve to plot. Either "ROC" or "PRC"

    fold_idx : int
        The index of the current fold, used for the plot legend

    plot_chance_level : bool
        Whether to plot the chance level for the curve

    Returns
    -------
    np.ndarray
        1D array with interpolated y-values based on mean_x, to be used for
        creating a mean eval curve

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
        name=f"{type} fold {fold_idx}",
        plot_chance_level=plot_chance_level,
        ax=ax,
        alpha=0.3,
        lw=1,
    )

    # interpolate results using mean_x param to allow for averaging of curves
    interp_y = np.interp(
        mean_x,
        config["interp_transform"](getattr(curve, config["x_attr"])),
        config["interp_transform"](getattr(curve, config["y_attr"])),
    )
    interp_y[0] = 0.0 if config["set_initial_interp_value"] else interp_y[0]

    return interp_y, getattr(curve, config["auc_attr"])


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
    y-values from cross validation folds. Also plots standard deviation bands
    around the mean curve.


    Parameters
    ----------
    mean_x : np.ndarray
        1D array with x values to be used to create a mean evaluation curve

    cv_y : np.ndarray
        Interpolated y-values for each cross validation fold, shape (F, N),
        where F is the number of folds and N is the number of examples in the
        validation set

    aucs : np.ndarray
        AUCs for each fold, shape (F,)

    ax : plt.Axes
        The axis object to plot the curve on

    plot_type : str
        Specifies the type of evaluation curve to plot. Either "ROC" or "PRC"

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
        title=config["tile"].format(class_name=class_name),
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
        % (type, auc_label, mean_auc, std_auc),
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

    return list(combined_probs.keys()), np.array(list(combined_probs.values()))
