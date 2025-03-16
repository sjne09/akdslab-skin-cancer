from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.core.groupby import DataFrameGroupBy
from scipy.stats import wilcoxon

from data_models.Label import Label


def signed_rank(
    grouped_data: DataFrameGroupBy, columns: List[str]
) -> pd.DataFrame:
    """
    Performs a paired wilcoxon signed rank test on each pair of groups
    in the DataFrameGroupBy object. Tests are one-sided with an
    alt hypothesis that the first group is greater than the second
    group.

    Parameters
    ----------
    grouped_data : DataFrameGroupBy
        The grouped data to perform the tests on. Tests will be
        performed on all (n choose 2) group pairings

    columns : List[str]
        The columns in the input data that contain the metrics to
        perform the tests on

    Returns
    -------
    pd.DataFrame
        The results of the paired wilcoxon signed rank tests
    """
    groups = list(grouped_data.groups.keys())
    results = []

    # perform signed rank test for all n choose 2 combinations
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            g1 = (
                grouped_data.get_group(groups[i])[columns]
                .to_numpy()
                .flatten()
                .astype(float)
            )
            g2 = (
                grouped_data.get_group(groups[j])[columns]
                .to_numpy()
                .flatten()
                .astype(float)
            )
            d = (g1 - g2).round(8)

            test_res = wilcoxon(d, alternative="greater")
            results.append(
                {
                    "group1": groups[i],
                    "group2": groups[j],
                    "test_stat": test_res.statistic,
                    "p_value": test_res.pvalue,
                }
            )
    results_df = pd.DataFrame.from_records(results)
    return results_df


def boxplots(
    grouped_data: DataFrameGroupBy,
    columns: List[str],
    title: str,
    save_name: str,
) -> None:
    """
    Generates boxplots for each group in a DataFrameGroupBy object and
    saves the final figure.

    Parameters
    ----------
    grouped_data : DataFrameGroupBy
        The grouped experiment data to create boxplots for. Data should
        be grouped by tile encoder/foundation model), slide encoder/
        aggregator, and classifier

    columns : List[str]
        The columns in the grouped dataframe containing the metrics to
        create the boxplots on

    title : str
        The figure title

    save_name : str
        The output path for the boxplot figure
    """
    groups = list(grouped_data.groups.keys())

    # first, flatten the data such that we have a vector of results for each
    # (model, aggregator, clf) combination; necessary because there are
    # multiple rows of data for each combo given crossval
    flattened_data = {}
    for group in groups:
        flattened_data[group] = (
            grouped_data.get_group(group)[columns]
            .to_numpy()
            .flatten()
            .astype(float)
        )
    flattened_df = pd.DataFrame(flattened_data)

    # create boxplots and save to disk
    fig, ax = plt.subplots(1, 1, figsize=(8, 15))
    ax.boxplot(flattened_df, tick_labels=flattened_df.columns, vert=False)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_name)


def signed_rank_summary(data: pd.DataFrame) -> pd.DataFrame:
    """
    Uses the results of paired wilcoxon signed rank tests to yield a
    summary of how many pairs a particular experiment performed greater
    than, equal to, or less than.

    Parameters
    ----------
    data : pd.DataFrame
        The wilcoxon signed rank test results

    Returns
    -------
    pd.DataFrame
        The summarized results
    """
    group_results = {}
    groups = np.unique(
        np.concat((data["group1"].unique(), data["group2"].unique()))
    )  # all tested combinations of (foundation model, aggregator, clf)
    for group in groups:
        group_results[group] = {
            "gt": 0,
            "eq": 0,
            "lt": 0,
        }

        # enumerate results where the current group is group1
        # since the signed rank tests are looking for a difference > 0, if
        # the p-value is <= 0.05, then group1 > group2 at an alpha of 0.05
        sub_df = data.loc[data["group1"] == group]
        total_tests = len(sub_df)
        group_results[group]["gt"] += len(
            sub_df.loc[sub_df["p_value"] <= 0.05]
        )
        group_results[group]["lt"] += len(
            sub_df.loc[sub_df["p_value"] >= 0.95]
        )

        # enumerate results where the current group is group2
        # group2 > group1 with probability 1 - p-value
        sub_df = data.loc[data["group2"] == group]
        total_tests += len(sub_df)
        group_results[group]["gt"] += len(
            sub_df.loc[sub_df["p_value"] >= 0.95]
        )
        group_results[group]["lt"] += len(
            sub_df.loc[sub_df["p_value"] <= 0.05]
        )

        # for remaining results, we accept the null that the difference b/w
        # groups is symmetric about 0
        group_results[group]["eq"] = (
            total_tests
            - group_results[group]["gt"]
            - group_results[group]["lt"]
        )

    # return the results in a dataframe
    return (
        pd.DataFrame.from_dict(group_results, orient="index")
        .reset_index()
        .rename(
            columns={
                "level_0": "foundation_model",
                "level_1": "aggregator",
                "level_2": "classifier",
            }
        )
    )


def inference_comparison(
    src1: str,
    src2: str,
    label_to_compare: int,
    model1_name: str,
    model2_name: str,
) -> None:
    """
    Compares the predictions of two models for a given class of interest based
    on the rank of the positive and negative instances. First saves a file
    for each model to disk, containing columns with the rank of each positive
    and negative instance for the class of interest. Then saves a file to
    disk containing the difference in the proportions for each slide between
    the two models.

    Parameters
    ----------
    src1 : str
        The path to the predictions of the first model. Should be s csv with
        columns "id", "ground_truth", and the class labels

    src2 : str
        The path to the predictions of the second model. Should be s csv with
        columns "id", "ground_truth", and the class labels

    label_to_compare : int
        The class of interest to compare the models on. Should be one of the
        class labels from the Label enum

    model1_name : str
        The name of the first model to use in the output filename

    model2_name : str
        The name of the second model to use in the output filename
    """
    label_col = Label(label_to_compare).name
    relevant_cols = ["ground_truth", label_col]

    # load the predictions from the two models and get the ranks
    model1_preds = pd.read_csv(src1, index_col=0)[relevant_cols].sort_values(
        by=label_col, ascending=False
    )
    model2_preds = pd.read_csv(src2, index_col=0)[relevant_cols].sort_values(
        by=label_col, ascending=False
    )

    _get_ranks(model1_preds, label_to_compare, label_col)
    _get_ranks(model2_preds, label_to_compare, label_col)

    # save the results to disk
    model1_preds.to_csv(f"outputs/{model1_name}-{label_col}.csv")
    model2_preds.to_csv(f"outputs/{model2_name}-{label_col}.csv")

    # compare the results and save to disk
    comparison_df = pd.merge(
        model1_preds[["rank_of_pos", "rank_of_neg"]],
        model2_preds[["rank_of_pos", "rank_of_neg"]],
        left_index=True,
        right_index=True,
        suffixes=(f"_{model1_name}", f"_{model2_name}"),
    )
    comparison_df["rank_of_pos_diff"] = (
        comparison_df[f"rank_of_pos_{model1_name}"]
        - comparison_df[f"rank_of_pos_{model2_name}"]
    ).abs()
    comparison_df["rank_of_neg_diff"] = (
        comparison_df[f"rank_of_neg_{model1_name}"]
        - comparison_df[f"rank_of_neg_{model2_name}"]
    ).abs()
    comparison_df.sort_values(
        by=["rank_of_pos_diff", "rank_of_neg_diff"], ascending=False
    ).to_csv(
        "outputs/model_comparisons/"
        f"{model1_name}-{model2_name}-{label_col}.csv"
    )


def _get_pos_neg_counts(
    df: pd.DataFrame, label_to_compare: int
) -> Tuple[int, int]:
    """
    Given a dataframe containing the predictions of a model, returns the
    number of positive and negative instances for the class of interest.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the predictions of the model

    label_to_compare : int
        The class of interest to compare the models on. Should be one of the
        class labels from the Label enum

    Returns
    -------
    Tuple[int, int]
        A tuple containing the number of positive and negative instances for
        the class of interest
    """
    pos = df.loc[df["ground_truth"] == label_to_compare].shape[0]
    neg = df.loc[df["ground_truth"] != label_to_compare].shape[0]
    return pos, neg


def _get_ranks(
    df: pd.DataFrame, label_to_compare: int, label_col: str
) -> None:
    """
    Given a dataframe containing the predictions of a model, calculates the
    ranks of the positive and negative instances for the class of interest
    and stores them in the dataframe. The dataframe is modified in place.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the predictions of the model

    label_to_compare : int
        The class of interest to compare the models on. Should be one of the
        class labels from the Label enum

    label_col : str
        The name of the column in the dataframe containing the predictions
        for the class of interest
    """
    pos, neg = _get_pos_neg_counts(df, label_to_compare)

    def rank_of_pos_instance(row, df):
        """
        Counts the number of negative instances (in terms of the class of
        interest) that the model predicted with a lower score for that class
        than the current row, given that the current row is a positive
        instance. The rank is normalized by the total number of negative
        instances.
        """
        if row["ground_truth"] == label_to_compare:
            return (
                df.loc[
                    (df["ground_truth"] != label_to_compare)
                    & (df[label_col] < row[label_col])
                ].shape[0]
                / neg
            )
        else:
            return -1

    def rank_of_neg_instance(row, df):
        """
        Counts the number of positive instances (in terms of the class of
        interest) that the model predicted with a higher score for that class
        than the current row, given that the current row is a negative
        instance. The rank is normalized by the total number of negative
        instances.
        """
        if row["ground_truth"] != label_to_compare:
            return (
                df.loc[
                    (df["ground_truth"] == label_to_compare)
                    & (df[label_col] > row[label_col])
                ].shape[0]
                / pos
            )
        else:
            return -1

    df["rank_of_pos"] = df.apply(rank_of_pos_instance, axis=1, df=df)
    df["rank_of_neg"] = df.apply(rank_of_neg_instance, axis=1, df=df)
