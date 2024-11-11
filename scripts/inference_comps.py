import os

from dotenv import load_dotenv

from data_models.Label import Label
from utils.compare_results import inference_comparison

load_dotenv()

OUTPUT_DIR = os.getenv("OUTPUT_DIR")

# define paths to the preds for the models to be compared
models = ["prism", "uni", "gigapath"]
model_pred_paths = [
    "prism/preds/prism-PRISM-fold-3.csv",
    "uni/preds/uni-abmil-1_head-fold-3.csv",
    "gigapath/preds/gigapath-abmil-1_head-fold-1.csv",
]

for i in range(len(models) - 1):
    for j in range(i, len(models)):
        for label in Label:
            inference_comparison(
                os.path.join(OUTPUT_DIR, model_pred_paths[i]),
                os.path.join(OUTPUT_DIR, model_pred_paths[j]),
                label.value,
                models[i],
                models[j],
            )
