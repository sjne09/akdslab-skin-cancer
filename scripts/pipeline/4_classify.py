import os
from functools import partial
from typing import Any, Dict, List

import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.optim import AdamW
from xgboost import XGBClassifier

from data_processing.data_utils import get_slides_by_specimen
from data_processing.datasets import (
    SlideClassificationDataset,
    collate_slide_embeds,
)
from data_processing.label import Label
from data_processing.specimen_data import SpecimenData
from data_processing.split import train_val_split_sk_clf
from evaluation.eval import Evaluator
from models import MLP
from models.training.load import get_loaders
from models.training.train import Trainer

DATA_DIR = os.environ["DATA_DIR"]
OUTPUT_DIR = os.environ["OUTPUT_DIR"]
EPOCHS = 30
BATCH_SIZE = 16
NUM_LABELS = len(Label)
PATIENCE = 10


def crossval_mlp(
    data: SpecimenData,
    slides_by_specimen: Dict[str, List[str]],
    embedding_path: str,
    device: torch.device,
    foundation_model: str,
    aggregator: str,
) -> None:
    """
    Performs cross-validation for an MLP classifier on slide embeddings.

    Parameters
    ----------
    data : SpecimenData
        The data to use for training and validation

    slides_by_specimen : Dict[str, List[str]]
        The slides associated with each specimen, with the specimen ids
        as keys and the slide ids as values

    embedding_path : str
        The path to the slide embeddings

    device : torch.device
        The device to use for training

    foundation_model : str
        The foundation model used to generate the tile embeddings

    aggregator : str
        The aggregation /model used to generate the slide embeddings
    """
    model_name = (
        "chkpts/{foundation_model}/"
        "{foundation_model}-{aggregator}-fold-{i}.pt"
    )
    exp_name = f"{foundation_model}/{foundation_model}-MLP"

    evaluator = Evaluator(
        labels=Label,
        foundation_model=foundation_model,
        aggregator=aggregator,
        classifier="MLP",
        onehot_labels=data.onehot_labels,
    )

    trainer = Trainer(
        model_name_pattern=os.path.join(OUTPUT_DIR, model_name),
        evaluator=evaluator,
        device=device,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        patience=PATIENCE,
        n_folds=data.n_folds,
        input_keys=["slide_embed"],
        label_key="label",
    )

    for i in range(data.n_folds):
        print(f"--------------------FOLD    {i + 1}--------------------")
        # get dataloaders and embed dims for loaded embeddings
        train_loader, val_loader = get_loaders(
            val_fold=i,
            specimens_by_fold=data.specimens_by_fold,
            slides_by_specimen=slides_by_specimen,
            labels_by_specimen=data.labels,
            train_dataset_class=partial(
                SlideClassificationDataset, slide_embeds_path=embedding_path
            ),
            val_dataset_class=partial(
                SlideClassificationDataset, slide_embeds_path=embedding_path
            ),
            collate_fn=collate_slide_embeds,
        )
        sample = next(iter(train_loader))
        embed_dim = sample["slide_embed"].shape[-1]

        # initialize model, loss function, and optimizer
        model = MLP(embed_dim, [1024, 512, 256], NUM_LABELS).to(device)
        loss_fn = nn.CrossEntropyLoss()
        optim = AdamW(model.parameters(), lr=1e-5)

        trainer.training_loop(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optim=optim,
            loss_fn=loss_fn,
            save_name_args={
                "foundation_model": foundation_model,
                "aggregator": aggregator,
                "i": i,
            },
        )

    # get and save final outputs
    evaluator.finalize(data.class_freqs)
    evaluator.save_figs(exp_name)
    evaluator.results.to_csv(
        "outputs/experiments_by_fold.csv", sep="|", mode="a", header=False
    )


def crossval_sklearn(
    clf: Any,
    data: SpecimenData,
    slides_by_specimen: Dict[str, List[str]],
    exp_name: str,
    foundation_model: str,
    aggregator: str,
    classifier: str,
    embedding_path: str,
) -> pd.DataFrame:
    """
    Performs cross-validation for a scikit-learn-style classifier on
    slide embeddings.

    Parameters
    ----------
    clf : Any
        The classifier to use. Must have a `fit` and `predict_proba`
        method

    data : SpecimenData
        The data to use for training and validation

    slides_by_specimen : Dict[str, List[str]]
        The slides associated with each specimen, with the specimen ids
        as keys and the slide ids as values

    exp_name : str
        The name of the experiment

    foundation_model : str
        The foundation model used to generate the tile embeddings

    aggregator : str
        The aggregation method/model used to generate the slide
        embeddings

    classifier : str
        The name of the classifier
    """
    evaluator = Evaluator(
        labels=Label,
        foundation_model=foundation_model,
        aggregator=aggregator,
        classifier=classifier,
        onehot_labels=data.onehot_labels,
    )

    for i in range(data.n_folds):
        # fit the classifier on the train data and extract probs
        split = train_val_split_sk_clf(
            val_fold=i,
            specimens_by_fold=data.specimens_by_fold,
            slides_by_specimen=slides_by_specimen,
            labels_by_specimen=data.labels,
            embedding_path=embedding_path,
        )

        clf.fit(split["X_train"], split["y_train"])
        probs = clf.predict_proba(split["X_val"])
        model_data = {
            "ids": split["val_ids"],
            "labels": split["y_val"],
            "probs": probs,
        }

        evaluator.fold(i, model_data, data.n_folds)

    evaluator.finalize(data.class_freqs)
    evaluator.save_figs(exp_name)
    evaluator.results.to_csv(
        "outputs/experiments_by_fold.csv", sep="|", mode="a", header=False
    )


def main() -> None:
    """
    Runs the classification experiments for slide embeds from
    pre-trained models.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    label_path = os.path.join(DATA_DIR, "labels/labels.csv")
    fold_path = os.path.join(DATA_DIR, "folds.json")
    data = SpecimenData(label_path=label_path, fold_path=fold_path)

    slide_ids = [
        os.path.splitext(os.path.basename(path))[0]
        for path in os.listdir(os.path.join(OUTPUT_DIR, "uni/tile_embeddings"))
    ]
    slides_by_specimen = get_slides_by_specimen(slide_ids)

    slide_embeds = {
        "prism": ["GAP", "perceiver"],
        "uni": ["GAP"],
        "gigapath": ["GAP", "pool", "cls"],
        "resnet18": ["GAP"],
    }

    for fm, aggs in slide_embeds.items():
        for agg in aggs:
            embedding_path = os.path.join(
                OUTPUT_DIR,
                f"{fm}/slide_embeddings/{fm}_slide_embeds_{agg}.pkl",
            )

            crossval_mlp(
                data=data,
                slides_by_specimen=slides_by_specimen,
                embedding_path=embedding_path,
                device=device,
                foundation_model=fm,
                aggregator=agg,
            )

            crossval_sklearn(
                clf=XGBClassifier(
                    objective="multi:softmax", num_class=NUM_LABELS
                ),
                data=data,
                slides_by_specimen=slides_by_specimen,
                exp_name=f"{fm}/{agg}/{fm}-xgb",
                foundation_model=fm,
                aggregator=agg,
                classifier="xgb",
                embedding_path=embedding_path,
            )

            crossval_sklearn(
                clf=LogisticRegression(max_iter=1000, solver="saga"),
                data=data,
                slides_by_specimen=slides_by_specimen,
                exp_name=f"{fm}/{agg}/{fm}-lr",
                foundation_model=fm,
                aggregator=agg,
                classifier="lr",
                embedding_path=embedding_path,
            )


if __name__ == "__main__":
    main()
