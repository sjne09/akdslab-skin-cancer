from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.core.groupby import DataFrameGroupBy
from scipy.stats import wilcoxon


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


def main():
    auroc_keys = [k + "_auroc" for k in ["benign", "bowens", "bcc", "scc"]]
    auprc_keys = [k + "_auprc" for k in ["benign", "bowens", "bcc", "scc"]]

    data = pd.read_csv("outputs/experiments_by_fold.csv", sep="|", index_col=0)
    data = data.sort_values(by="fold", axis=0)
    grouped = data.groupby(by=["foundation_model", "aggregator", "classifier"])

    signed_rank_res = signed_rank(grouped, auroc_keys + auprc_keys)
    signed_rank_res.to_csv("outputs/signed_rank_results.csv")
    summary = signed_rank_summary(signed_rank_res).sort_values(
        "gt", ascending=False
    )
    summary.to_csv("outputs/signed_rank_summary.csv")

    boxplots(
        grouped, auroc_keys, "AUROC boxplots", "outputs/auroc_boxplot.png"
    )
    boxplots(grouped, auroc_keys, "AP boxplots", "outputs/ap_boxplot.png")


if __name__ == "__main__":
    main()
