from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.core.groupby import DataFrameGroupBy
from scipy.stats import wilcoxon

from data_models.Label import Label


def signed_rank(
    grouped_data: DataFrameGroupBy, columns: List[str]
) -> pd.DataFrame:
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
    groups = list(grouped_data.groups.keys())

    # first, flatten the data such that we have a vector of results for each
    # (model, aggregator, clf) combination
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
    Compares the predictions of two models for a given class of interest.
    First saves a file for each model to disk, containing columns with
    the following:
        - If a slide is positive for the class of interest, the proportion of
          negative instances that the model predicted with a lower score for
          the class of interest
        - If a slide is negative for the class of interest, the proportion of
          positive instances that the model predicted with a higher score for
          the class of interest
    Then saves a file to disk containing the difference in the proportions for
    each slide between the two models.

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

    # load the predictions from the two models
    model1_preds = pd.read_csv(src1, index_col=0)[relevant_cols].sort_values(
        by=label_col, ascending=False
    )
    model2_preds = pd.read_csv(src2, index_col=0)[relevant_cols].sort_values(
        by=label_col, ascending=False
    )

    # get the number of positive and negative instances for the class of
    # interest
    total_positives = model1_preds.loc[
        model1_preds["ground_truth"] == label_to_compare
    ].shape[0]
    total_negatives = model1_preds.loc[
        model1_preds["ground_truth"] != label_to_compare
    ].shape[0]

    # ensure that both dataframes contain the same number of instances of
    # positive and negative instances for the class of interest
    assert (
        total_positives
        == model2_preds.loc[
            model2_preds["ground_truth"] == label_to_compare
        ].shape[0]
    )
    assert (
        total_negatives
        == model2_preds.loc[
            model2_preds["ground_truth"] != label_to_compare
        ].shape[0]
    )

    def get_negative_count(row, df):
        """
        Counts the number of negative instances (in terms of the class of
        interest) that the model predicted with a lower score for that class
        than the current row, given that the current row is a positive
        instance.
        """
        if row["ground_truth"] == label_to_compare:
            return (
                df.loc[
                    (df["ground_truth"] != label_to_compare)
                    & (df[label_col] < row[label_col])
                ].shape[0]
                / total_negatives
            )
        else:
            return -1

    def get_positive_count(row, df):
        """
        Counts the number of positive instances (in terms of the class of
        interest) that the model predicted with a higher score for that class
        than the current row, given that the current row is a negative
        instance.
        """
        if row["ground_truth"] != label_to_compare:
            return (
                df.loc[
                    (df["ground_truth"] == label_to_compare)
                    & (df[label_col] > row[label_col])
                ].shape[0]
                / total_positives
            )
        else:
            return -1

    # get counts for each model
    model1_preds["negative_count"] = model1_preds.apply(
        get_negative_count, axis=1, df=model1_preds
    )
    model1_preds["positive_count"] = model1_preds.apply(
        get_positive_count, axis=1, df=model1_preds
    )
    model2_preds["negative_count"] = model2_preds.apply(
        get_negative_count, axis=1, df=model2_preds
    )
    model2_preds["positive_count"] = model2_preds.apply(
        get_positive_count, axis=1, df=model2_preds
    )

    # save the results to disk
    model1_preds.to_csv(f"outputs/{model1_name}-{label_col}.csv")
    model2_preds.to_csv(f"outputs/{model2_name}-{label_col}.csv")

    # compare the results and save to disk
    comparison_df = pd.merge(
        model1_preds[["negative_count", "positive_count"]],
        model2_preds[["negative_count", "positive_count"]],
        left_index=True,
        right_index=True,
        suffixes=("_prism", "_uni"),
    )
    comparison_df["negative_diff"] = (
        comparison_df["negative_count_prism"]
        - comparison_df["negative_count_uni"]
    ).abs()
    comparison_df["positive_diff"] = (
        comparison_df["positive_count_prism"]
        - comparison_df["positive_count_uni"]
    ).abs()
    comparison_df.sort_values(
        by=["negative_diff", "positive_diff"], ascending=False
    ).to_csv(f"outputs/{model1_name}-{model2_name}-{label_col}.csv")
