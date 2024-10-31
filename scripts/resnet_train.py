import os

import torch
from dotenv import load_dotenv
from torch import nn
from torch.optim import AdamW
from torchvision import transforms

from data_models.datasets import TileLoaderDataset
from models.agg import MILClassifier
from models.resnet import ResNetFeatureExtractor
from utils.load_data import SpecimenData
from utils.split import train_val_split_labels, train_val_split_slides
from utils.tile_embed_postproc import add_positions

load_dotenv()
DATA_DIR = os.getenv("DATA_DIR")
OUTPUT_DIR = os.getenv("OUTPUT_DIR")

gpus = ["0"]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    # TODO: train tranform with randomness
    transform = transforms.Compose(
        [
            transforms.Resize(
                256, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
        ]
    )

    # EPOCHS = 30
    BATCH_SIZE = 4
    NUM_LABELS = 4
    # PATIENCE = 10

    # for each fold
    for i in range(len(slides_by_specimen)):
        print(f"--------------------FOLD    {i + 1}--------------------")

        # load model and optimizer for end-to-end training
        resnet = ResNetFeatureExtractor()
        model = MILClassifier(resnet.embed_dim, NUM_LABELS, 1, False)
        resnet.to(device)
        model.to(device)

        loss_fn = nn.CrossEntropyLoss()
        optim = AdamW(
            list(resnet.parameters()) + list(model.parameters()), lr=1e-5
        )

        # best_loss = float("inf")
        # best_model_weights = None
        # best_model_data = {"ids": None, "labels": None, "probs": None}
        # patience = PATIENCE

        # get train/val split for slide ids and labels
        train_slides, val_slides = train_val_split_slides(
            i, data.specimens_by_fold, slides_by_specimen
        )
        train_labels, val_labels = train_val_split_labels(
            i, data.labels, data.specimens_by_fold
        )

        # get list of dirs containing tile images for each slide in train/val
        train_tile_dirs = [
            os.path.join(tile_img_dir, slide) for slide in train_slides
        ]
        # val_tile_dirs = [
        #     os.path.join(tile_img_dir, slide) for slide in val_slides
        # ]

        # TODO: include labels in dataset
        train_dataset = TileLoaderDataset(
            train_tile_dirs, train_labels, 32, 512, transform
        )
        # val_dataset = TileLoaderDataset(
        #     val_tile_dirs, val_labels, 32, 512, transform
        # )

        agg_loss = 0.0
        resnet.train()
        model.train()

        # train dataset contains slide dataloaders
        # TODO: shuffle?
        for j, (slide_loader, label) in enumerate(train_dataset):
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
            tile_embeds["tile_embeds"] = tile_embeds["tile_embeds"].unsqueeze(
                0
            )
            tile_embeds["pos"] = tile_embeds["pos"].unsqueeze(0)
            logits = model(
                tile_embeds["tile_embeds"].to(device),
                tile_embeds["pos"].to(device),
            )
            loss = loss_fn(logits, label.to(device))
            loss.backward()
            agg_loss += loss.item()

            # accumulate grad until grad_accum_steps is reached
            if (j + 1) % BATCH_SIZE == 0:
                optim.step()
                optim.zero_grad()

        # ensure no remaining accumulated grad
        if (j + 1) % BATCH_SIZE != 0:
            optim.step()
            optim.zero_grad()

        print(agg_loss / (i + 1))


if __name__ == "__main__":
    main()
