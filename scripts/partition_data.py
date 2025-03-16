import json
import os

import numpy as np
import pandas as pd

DATA_DIR = os.environ["DATA_DIR"]
NUM_FOLDS = 5

# load data, add label column based three label cols
df = pd.read_csv(os.path.join(DATA_DIR, "labels/labels.csv"))
df["specimenid"] = df["specimenid"].astype("string")
df["patientid"] = df["patientid"].astype("string")
df["label"] = pd.from_dummies(
    df[["bcc1", "scc1", "bowens1"]], default_category="na"
)

# map patient ids to specimen ids
pat_to_specs = {k: set() for k in df["patientid"].unique()}
unmatched = set()  # track specimens that are not in the label data

for slide in os.listdir(os.path.join(DATA_DIR, "tiles/output")):
    if slide.endswith(".svs"):
        specimen_id = slide[:6]
        res = df.loc[df["specimenid"] == specimen_id]
        if res.shape[0] == 0:
            unmatched.add(specimen_id)
        else:
            pat_to_specs[res["patientid"].item()].add(specimen_id)

print(f"specimens that will not be used due to missing labels: {unmatched}")

# create dataframe mapping patient ids to the mode of the label column
# necessary since patients may have multiple specimens, each with different
# labels
labels_by_patient = (
    df.groupby("patientid")["label"]
    .agg(lambda x: pd.Series.mode(x)[0])
    .to_frame()
)

# merge new dataframe with existing to add overall patient label as mode
# across specs
df = df.merge(
    labels_by_patient,
    left_on="patientid",
    right_index=True,
    suffixes=("_spec", "_pat"),
)


# group data by patient label for stratification
groups = df.groupby("label_pat")

# partition data based on patients; stratify based on label
folds = [[] for _ in range(NUM_FOLDS)]
for name, indices in groups.groups.items():
    # extract data for group and shuffle
    group_data = df.iloc[indices].sample(frac=1, random_state=13)

    # get all patients within the group, split into folds
    group_patients = group_data["patientid"].unique()
    group_folds = np.array_split(group_patients, NUM_FOLDS)

    # create random assignment order for group folds; necessary since last
    # fold will always have less than the other folds if len(group_patients)
    # is not divisible by NUM_FOLDS
    assignment_order = np.arange(NUM_FOLDS)
    np.random.shuffle(assignment_order)

    # add sampled patients to each fold
    for i, fold in enumerate(folds):
        fold.extend(group_folds[assignment_order[i]])

# get lists of specimens for each fold based on patient folds
spec_folds = [[] for _ in range(NUM_FOLDS)]
for i, fold in enumerate(folds):
    for pat in fold:
        spec_folds[i].extend(list(pat_to_specs[pat]))

with open(os.path.join(DATA_DIR, "folds.json"), "w") as f:
    json.dump(spec_folds, f)
