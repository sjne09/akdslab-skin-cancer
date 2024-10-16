from typing import List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim,
    loss_fn: nn.Module,
    batch_size: int,
    device: torch.device,
) -> float:
    agg_loss = 0.0

    model.train()
    for i, sample in enumerate(dataloader):
        embeds = sample["tile_embeds"]
        label = sample["label"]
        coords = sample["pos"]

        logits = model(embeds.to(device), coords.to(device))
        loss = loss_fn(logits, label.to(device))
        loss.backward()
        agg_loss += loss.item()

        # accumulate grad until batch size is reached
        if (i + 1) % batch_size == 0:
            optimizer.step()
            optimizer.zero_grad()

    # ensure no remaining accumulated grad
    if (i + 1) % batch_size != 0:
        optimizer.step()
        optimizer.zero_grad()

    return agg_loss / (i + 1)


def val_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> Tuple[float, torch.Tensor, torch.Tensor, List[str]]:
    agg_loss = 0.0
    outputs = []
    labels = []
    ids = []

    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(dataloader):
            embeds = sample["tile_embeds"]
            label = sample["label"]
            coords = sample["pos"]

            logits = model(embeds.to(device), coords.to(device))
            loss = loss_fn(logits, label.to(device))
            agg_loss += loss.item()

            outputs.append(torch.softmax(logits.detach().cpu(), dim=-1))
            labels.append(label)
            ids.extend(sample["id"])
    outputs = torch.cat(outputs)
    labels = torch.cat(labels)
    return agg_loss / (i + 1), labels, outputs, ids
