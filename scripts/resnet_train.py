import copy
import os

import numpy as np
import torch
from dotenv import load_dotenv
from torch import nn
from torch.optim import AdamW
from torchvision import transforms
from torchvision.transforms import v2

from data_models.datasets import TileLoaderDataset
from models.agg import MILClassifier
from models.resnet import ResNetFeatureExtractor
from utils.load_data import SpecimenData
from utils.split import train_val_split_labels, train_val_split_slides
from utils.tile_embed_postproc import add_positions

load_dotenv()
DATA_DIR = os.getenv("DATA_DIR")
OUTPUT_DIR = os.getenv("OUTPUT_DIR")

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_datasets(
    fold_idx,
    specimens_by_fold,
    slides_by_specimen,
    labels,
    tile_img_dir,
):
    # get train/val split for slide ids and labels
    train_slides, val_slides = train_val_split_slides(
        fold_idx, specimens_by_fold, slides_by_specimen
    )
    train_labels, val_labels = train_val_split_labels(
        fold_idx, labels, specimens_by_fold
    )

    # get list of dirs containing tile images for each slide in train/val
    train_tile_dirs = [
        os.path.join(tile_img_dir, slide) for slide in train_slides
    ]
    val_tile_dirs = [os.path.join(tile_img_dir, slide) for slide in val_slides]

    # train tranform with augmentations
    train_transform = v2.Compose(
        [
            v2.Resize(
                256, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            v2.CenterCrop(224),
            v2.RandomRotation(degrees=360),
            v2.ColorJitter(),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
        ]
    )

    val_transform = v2.Compose(
        [
            v2.Resize(
                256, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            v2.CenterCrop(224),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
        ]
    )

    train_dataset = TileLoaderDataset(
        train_tile_dirs, train_labels, 512, 512, train_transform
    )
    val_dataset = TileLoaderDataset(
        val_tile_dirs, val_labels, 512, 512, val_transform
    )
    return train_dataset, val_dataset


def main():
    # load the labels data with folds
    label_path = os.path.join(DATA_DIR, "labels/labels.csv")
    fold_path = os.path.join(DATA_DIR, "folds.json")
    data = SpecimenData(label_path, fold_path)

    tile_img_dir = os.path.join(DATA_DIR, "tiles/output")
    tile_dirs = os.listdir(tile_img_dir)

    # map specimens to slides
    slides_by_specimen = {spec: [] for spec in data.specimens}
    for slide in tile_dirs:
        slide_name = os.path.basename(slide)[:-4]
        spec = slide_name[:6]
        if slides_by_specimen.get(spec) is not None:
            slides_by_specimen[spec].append(slide)

    EPOCHS = 30
    BATCH_SIZE = 16
    NUM_LABELS = 4
    PATIENCE = 10

    # for each fold
    for i in range(len(slides_by_specimen)):
        print(f"--------------------FOLD    {i + 1}--------------------")

        train_dataset, val_dataset = get_datasets(
            i,
            data.specimens_by_fold,
            slides_by_specimen,
            data.labels,
            tile_img_dir,
        )

        # load model and optimizer for end-to-end training
        resnet = ResNetFeatureExtractor()
        model = MILClassifier(resnet.embed_dim, NUM_LABELS, 1, False)
        resnet.to(device)
        model.to(device)

        loss_fn = nn.CrossEntropyLoss()
        optim = AdamW(
            list(resnet.parameters()) + list(model.parameters()), lr=1e-5
        )

        best_loss = float("inf")
        best_model_weights = None
        best_model_data = {"ids": None, "labels": None, "probs": None}
        patience = PATIENCE

        for epoch in range(EPOCHS):
            # training
            train_loss = 0.0
            resnet.train()
            model.train()

            # shuffle the dataset
            sample_idxs = np.random.permutation(len(train_dataset))

            # for each sample in the dataset, run all tiles through resnet,
            # then concat results and run through abmil classifier
            for j, idx in enumerate(sample_idxs):
                if j % 100 == 0:
                    print(f"train batch {j}")
                slide_loader, label = train_dataset[idx]

                # accumulate tile embeddings accross sample of slide tiles
                tile_embeds = []
                for sample in slide_loader:
                    tile_embed = resnet(sample["img"].to(device))
                    if len(tile_embed.shape) == 1:
                        tile_embed = tile_embed.unsqueeze(0)
                    tile_embeds.append(
                        {"tile_embeds": tile_embed, "coords": sample["coords"]}
                    )

                # aggregate tile embeddings and add positional data
                tile_embeds = {
                    k: torch.cat([item[k] for item in tile_embeds])
                    for k in tile_embeds[0].keys()
                }
                add_positions(tile_embeds)
                tile_embeds["tile_embeds"] = tile_embeds[
                    "tile_embeds"
                ].unsqueeze(0)
                tile_embeds["pos"] = tile_embeds["pos"].unsqueeze(0)
                logits = model(
                    tile_embeds["tile_embeds"].to(device),
                    tile_embeds["pos"].to(device),
                )
                loss = loss_fn(logits, label.to(device))
                loss.backward()
                train_loss += loss.item()

                # accumulate grad until grad_accum_steps is reached
                if (j + 1) % BATCH_SIZE == 0:
                    optim.step()
                    optim.zero_grad()

            # ensure no remaining accumulated grad
            if (j + 1) % BATCH_SIZE != 0:
                optim.step()
                optim.zero_grad()

            train_loss = train_loss / (j + 1)

            # validation
            val_loss = 0.0
            outputs = []
            labels = []
            resnet.eval()
            model.eval()
            for k, (slide_loader, label) in enumerate(val_dataset):
                if k % 100 == 0:
                    print(f"eval batch {k}")
                with torch.no_grad():
                    tile_embeds = []
                    for sample in slide_loader:
                        tile_embed = resnet(sample["img"].to(device))
                        if len(tile_embed.shape) == 1:
                            tile_embed = tile_embed.unsqueeze(0)
                        tile_embeds.append(
                            {
                                "tile_embeds": tile_embed,
                                "coords": sample["coords"],
                            }
                        )

                    # aggregate tile embeddings and add positional data
                    tile_embeds = {
                        k: torch.cat([item[k] for item in tile_embeds])
                        for k in tile_embeds[0].keys()
                    }
                    add_positions(tile_embeds)
                    tile_embeds["tile_embeds"] = tile_embeds[
                        "tile_embeds"
                    ].unsqueeze(0)
                    tile_embeds["pos"] = tile_embeds["pos"].unsqueeze(0)
                    logits = model(
                        tile_embeds["tile_embeds"].to(device),
                        tile_embeds["pos"].to(device),
                    )
                    loss = loss_fn(logits, label.to(device))
                    val_loss += loss.item()

                    outputs.append(
                        torch.softmax(logits.detach().cpu(), dim=-1)
                    )
                    labels.append(label.cpu())
            outputs = torch.cat(outputs)
            labels = torch.cat(labels)
            val_loss = val_loss / (k + 1)

            if val_loss < best_loss:
                best_loss = val_loss
                best_model_weights = copy.deepcopy(model.state_dict())
                best_model_data["labels"] = labels
                best_model_data["probs"] = outputs
                patience = PATIENCE
            else:
                patience -= 1
                if patience == 0:
                    break

            if (epoch + 1) % 2 == 0:
                spaces = " " * (4 - len(str(epoch + 1)))
                print(
                    "--------------------"
                    + f"EPOCH{spaces}{epoch + 1}"
                    + "--------------------"
                )
                print(f"train loss: {train_loss:0.6f}")
                print(f"val loss:   {val_loss:0.6f}")
                print()

        # save the best model
        torch.save(
            best_model_weights,
            os.path.join(OUTPUT_DIR, f"resnet_tuned-fold_{i}.pt"),
        )


if __name__ == "__main__":
    main()
