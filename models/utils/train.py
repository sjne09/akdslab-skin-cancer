from operator import itemgetter
from typing import List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim,
    loss_fn: nn.Module,
    grad_accum_steps: int,
    input_keys: List[str],
    label_key: str,
    device: Optional[torch.device] = None,
) -> float:
    """
    Trains an epoch.

    Parameters
    ----------
    model : nn.Module
        The model to train

    dataloader : DataLoader
        The dataloader for the training data

    optimizer : torch.optim
        The optimizer

    loss_fn : nn.Module
        The loss function

    grad_accum_steps : int
        The number of batches/samples to accumulate gradients for before
        updating the model

    input_keys : List[str]
        The keys for samples drawn from the dataloader containing data to use
        as input to the model

    label_key : str
        The key for samples drawn from the dataloader containing the labels
        for the model

    device : Optional[torch.device]
        The device to send the model and data to. If not provided, the model
        and data will not be moved to a device

    Returns
    -------
    float
        The average training loss for the epoch
    """
    agg_loss = 0.0

    model = model.to(device) if device else model
    model.train()
    for i, sample in enumerate(dataloader):
        # get the model inputs from the sample, convert to tuple if not
        # already; unsqueeze to add batch dimension if necessary
        model_input = itemgetter(*input_keys)(sample)
        model_input: Tuple[torch.Tensor] = (
            (model_input)
            if isinstance(model_input, torch.Tensor)
            else model_input
        )
        model_input = [
            item.unsqueeze(0) for item in model_input if len(item.shape) == 1
        ]
        label: torch.Tensor = sample[label_key]

        logits = model(*[item.to(device) for item in model_input])
        loss = loss_fn(logits, label.to(logits.device))
        loss.backward()
        agg_loss += loss.item()

        # accumulate grad until grad_accum_steps is reached
        if (i + 1) % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

    # ensure no remaining accumulated grad
    if (i + 1) % grad_accum_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    return agg_loss / (i + 1)


def val_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    input_keys: List[str],
    label_key: str,
    loss_fn: Optional[nn.Module] = None,
) -> Tuple[float, torch.Tensor, torch.Tensor, List[str]]:
    """
    Validates an epoch.

    Parameters
    ----------
    model : nn.Module
        The model to validate

    dataloader : DataLoader
        The dataloader for the validation data

    device : torch.device
        The device to send the model and data to

    input_keys : List[str]
        The keys for samples drawn from the dataloader containing data to use
        as input to the model

    label_key : str
        The key for samples drawn from the dataloader containing the labels
        for the model

    loss_fn : nn.Module, optional
        The loss function; if not provided no loss will be calculated

    Returns
    -------
    float
        The average training loss for the epoch; if loss_fn is not provided
        this will be 0

    torch.Tensor
        The labels for the validation data

    torch.Tensor
        The model outputs for the validation data (softmaxed logits)

    List[str]
        The IDs for the validation data
    """
    agg_loss = 0.0
    outputs = []
    labels = []
    ids = []

    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(dataloader):
            # get the model inputs from the sample, convert to tuple if not
            # already; unsqueeze to add batch dimension if necessary
            model_input = itemgetter(*input_keys)(sample)
            model_input: Tuple[torch.Tensor] = (
                (model_input)
                if isinstance(model_input, torch.Tensor)
                else model_input
            )
            model_input = [
                item.unsqueeze(0) if len(item.shape) == 1 else item
                for item in model_input
            ]
            label: torch.Tensor = sample[label_key]

            logits = model(*[item.to(device) for item in model_input])
            if loss_fn is not None:
                loss = loss_fn(logits, label.to(device))
                agg_loss += loss.item()

            outputs.append(torch.softmax(logits.detach().cpu(), dim=-1))
            labels.append(label)
            ids.extend(sample["id"])
    outputs = torch.cat(outputs)
    labels = torch.cat(labels)
    return agg_loss / (i + 1), labels, outputs, ids
