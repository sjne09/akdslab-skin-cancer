import os

from data_processing.label import Label
from evaluation.compare_results import inference_comparison

OUTPUT_DIR = os.environ["OUTPUT_DIR"]

# define paths to the preds for the models to be compared
models = ["prism", "uni", "gigapath"]
model_pred_paths = [
    "prism/preds/prism-PRISM-fold-3.csv",
    "uni/preds/uni-abmil-1_head-fold-3.csv",
    "gigapath/preds/gigapath-abmil-1_head-fold-1.csv",
]

for i in range(len(models) - 1):
    for j in range(i + 1, len(models)):
        print(models[i], models[j])
        for label in Label:
            inference_comparison(
                os.path.join(OUTPUT_DIR, model_pred_paths[i]),
                os.path.join(OUTPUT_DIR, model_pred_paths[j]),
                label.value,
                models[i],
                models[j],
            )
