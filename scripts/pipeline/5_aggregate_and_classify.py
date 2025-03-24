import os
from functools import partial
from typing import Dict, List

import torch
from torch import nn
from torch.optim import AdamW

from data_processing.data_utils import get_slides_by_specimen
from data_processing.datasets import SlideEncodingDataset, collate_tile_embeds
from data_processing.label import Label
from data_processing.specimen_data import SpecimenData
from evaluation.eval import Evaluator
from models import MILClassifier
from models.training.load import get_loaders
from models.training.train import Trainer

DATA_DIR = os.getenv("DATA_DIR")
OUTPUT_DIR = os.getenv("OUTPUT_DIR")
EPOCHS = 30
BATCH_SIZE = 16
NUM_LABELS = len(Label)
PATIENCE = 10


def crossval(
    data: SpecimenData,
    slides_by_specimen: Dict[str, List[str]],
    embedding_dir: str,
    device: torch.device,
    foundation_model: str,
    heads: int,
    gated: bool,
) -> None:
    """
    Performs cross-validation for ABMIL + MLP classifier.

    Parameters
    ----------
    data : SpecimenData
        The data to use for training and validation

    slides_by_specimen : Dict[str, List[str]]
        The slides associated with each specimen, with the specimen ids
        as keys and the slide ids as values

    embedding_dir : str
        The path to the tile embeddings directory

    device : torch.device
        The device to use for training

    foundation_model : str
        The foundation model used to generate the tile embeddings

    heads : int
        The number of attention heads to use in the ABMIL model

    gated : bool
        Whether to use the gated or ungated ABMIL model
    """
    aggregator = "gabmil" if gated else "abmil"
    save_directory = "gated" if gated else "ungated"
    model_name = (
        "chkpts/{foundation_model}-{aggregator}-{heads}_heads-fold-{i}.pt"
    )
    exp_name = (
        f"{foundation_model}/abmil/{save_directory}/"
        f"{foundation_model}-3_hidden-{heads}_heads"
    )

    evaluator = Evaluator(
        labels=Label,
        foundation_model=foundation_model,
        aggregator=aggregator,
        classifier="MLP",
        onehot_labels=data.onehot_labels,
    )

    trainer = Trainer(
        model_name_pattern=model_name,
        evaluator=evaluator,
        device=device,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        patience=PATIENCE,
        n_folds=data.n_folds,
        input_keys=["tile_embeds", "pos"],
        label_key="label",
    )

    # for each fold, train and validate while saving the best models
    # then evaluate the final outputs by generating curves using Evaluator
    for i in data.n_folds:
        print(f"--------------------FOLD    {i + 1}--------------------")
        # get dataloaders and embed dims for loaded embeddings
        train_loader, val_loader = get_loaders(
            val_fold=i,
            specimens_by_fold=data.specimens_by_fold,
            slides_by_specimen=slides_by_specimen,
            labels_by_specimen=data.labels,
            train_dataset_class=partial(
                SlideEncodingDataset.from_slide_ids,
                tile_embed_dir=embedding_dir,
            ),
            val_dataset_class=partial(
                SlideEncodingDataset.from_slide_ids,
                tile_embed_dir=embedding_dir,
            ),
            collate_fn=collate_tile_embeds,
        )
        sample = next(iter(train_loader))
        embed_dim = sample["tile_embeds"].shape[-1]

        # initialize model, loss function, and optimizer
        model = MILClassifier(embed_dim, NUM_LABELS, heads, gated).to(device)
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
                "heads": heads,
                "i": i,
            },
        )

    evaluator.finalize(data.class_freqs)
    evaluator.save_figs(exp_name)
    evaluator.results.to_csv(
        "outputs/experiments_by_fold.csv", sep="|", mode="a", header=False
    )


def main() -> None:
    """
    Runs the cross-validation experiments for the ABMIL + MLP
    aggregator + classifier.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    label_path = os.path.join(DATA_DIR, "labels/labels.csv")
    fold_path = os.path.join(DATA_DIR, "folds.json")
    data = SpecimenData(label_path=label_path, fold_path=fold_path)

    slide_ids = [
        os.path.splitext(os.path.basename(path))[0]
        for path in os.listdir(
            os.path.join(OUTPUT_DIR, "uni/tile_embeddings_sorted")
        )
    ]
    slides_by_specimen = get_slides_by_specimen(slide_ids)

    fms = ["uni", "gigapath", "prism", "resnet18"]
    gates = [False, False, True, True]
    head_counts = [1, 8, 1, 8]

    for foundation_model in fms:
        for exp_idx in range(len(gates)):
            print(f"\n\n****EXPERIMENT {exp_idx + 1}****\n\n")
            # set experiment-specific variables
            gated = gates[exp_idx]
            heads = head_counts[exp_idx]

            embedding_dir = (
                "/opt/gpudata/skin-cancer/outputs/"
                f"{foundation_model}/tile_embeddings_sorted"
            )

            crossval(
                data=data,
                slides_by_specimen=slides_by_specimen,
                embedding_dir=embedding_dir,
                device=device,
                foundation_model=foundation_model,
                heads=heads,
                gated=gated,
            )
