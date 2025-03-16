import pandas as pd

from evaluation.compare_results import (
    boxplots,
    signed_rank,
    signed_rank_summary,
)

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

boxplots(grouped, auroc_keys, "AUROC boxplots", "outputs/auroc_boxplot.png")
boxplots(grouped, auroc_keys, "AP boxplots", "outputs/ap_boxplot.png")
