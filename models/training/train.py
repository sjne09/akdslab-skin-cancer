from copy import deepcopy
from operator import itemgetter
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from evaluation.eval import Evaluator


class Trainer:
    def __init__(
        self,
        model_name_pattern: str,
        evaluator: Evaluator,
        device: torch.device,
        epochs: int,
        batch_size: int,
        patience: int,
        n_folds: int,
        input_keys: List[str],
        label_key: str,
    ) -> None:
        """
        Parameters
        ----------
        model_name_pattern : str
            The pattern for the model name - should be a path with
            placeholders for key information

        evaluator : Evaluator
            The evaluator to use for evaluation

        device : torch.device
            The device to use for training

        epochs : int
            The number of epochs to train for

        batch_size : int
            The batch size to use for training. Uses grad accumulation
            rather than batching

        patience : int
            The number of epochs to wait before early stopping

        n_folds : int
            The number of folds to train

        input_keys : List[str]
            The keys for samples drawn from the dataloader containing
            data to use as input to the model

        label_key : str
            The key for samples drawn from the dataloader containing
            the ground truth labels for the data
        """
        self.model_name_pattern = model_name_pattern
        self.evaluator = evaluator
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.n_folds = n_folds
        self.input_keys = input_keys
        self.label_key = label_key
        self.current_fold = 0

    def training_loop(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optim: torch.optim,
        loss_fn: nn.Module,
        save_name_args: Dict[str, Any],
    ) -> None:
        """
        Performs the training loop for the model.

        Parameters
        ----------
        model : nn.Module
            The model to train

        train_loader : DataLoader
            The dataloader for the training data

        val_loader : DataLoader
            The dataloader for the validation data

        optim : torch.optim
            The optimizer to use

        loss_fn : nn.Module
            The loss function to use

        save_name_args : Dict[str, Any]
            The arguments to pass to the model_name_pattern
        """
        best_loss = float("inf")
        best_model_weights = None
        best_model_data = {"ids": None, "labels": None, "probs": None}
        patience = self.patience

        for epoch in range(self.epochs):
            train_loss = self.train_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optim,
                loss_fn=loss_fn,
                grad_accum_steps=self.batch_size,
                input_keys=self.input_keys,
                label_key=self.label_key,
                device=self.device,
            )
            val_loss, labels, probs, ids = self.val_epoch(
                model=model,
                dataloader=val_loader,
                loss_fn=loss_fn,
                input_keys=self.input_keys,
                label_key=self.label_key,
                device=self.device,
            )

            if val_loss < best_loss:
                best_loss = val_loss
                best_model_weights = deepcopy(model.state_dict())
                best_model_data["ids"] = ids
                best_model_data["labels"] = labels
                best_model_data["probs"] = probs
                patience = self.patience
            else:
                patience -= 1
                if patience == 0:
                    break

            if (epoch + 1) % 2 == 0:
                spaces = " " * (4 - len(str(epoch + 1)))
                print(
                    "--------------------"
                    f"EPOCH{spaces}{epoch + 1}"
                    "--------------------"
                )
                print(f"train loss: {train_loss:0.6f}")
                print(f"val loss:   {val_loss:0.6f}")
                print()

        # save the best model
        self.save_model(best_model_weights, save_name_args)
        self.evaluator.fold(self.fold, best_model_data, self.n_folds)
        self.fold += 1

    def save_model(
        self, model_state_dict: Dict[str, Any], save_name_args: Dict[str, Any]
    ) -> None:
        """
        Saves the model state dict to the path specified by the
        model_name_pattern.

        Parameters
        ----------
        model_state_dict : Dict[str, Any]
            The model state dict to save

        save_name_args : Dict[str, Any]
            The arguments to pass to the model_name_pattern
        """
        torch.save(
            model_state_dict, self.model_name_pattern.format(**save_name_args)
        )

    @staticmethod
    def train_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim,
        loss_fn: nn.Module,
        grad_accum_steps: int,
        input_keys: List[str],
        label_key: str,
        device: torch.device,
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
            The number of batches/samples to accumulate gradients for
            before updating the model

        input_keys : List[str]
            The keys for samples drawn from the dataloader containing
            data to use as input to the model

        label_key : str
            The key for samples drawn from the dataloader containing
            the labels for the model

        device : torch.device
            The device to send the model and data to. If not provided,
            the model and data will not be moved to a device

        Returns
        -------
        float
            The average training loss for the epoch
        """
        agg_loss = 0.0

        model = model.to(device)
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
                item.unsqueeze(0) if len(item.shape) == 1 else item
                for item in model_input
            ]
            label: torch.Tensor = sample[label_key]

            logits = model(*[item.to(device) for item in model_input])
            logits = logits[0] if isinstance(logits, tuple) else logits

            loss = loss_fn(logits, label.to(device))
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

    @staticmethod
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
            The keys for samples drawn from the dataloader containing
            data to use as input to the model

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
                logits = logits[0] if isinstance(logits, tuple) else logits

                if loss_fn is not None:
                    loss = loss_fn(logits, label.to(device))
                    agg_loss += loss.item()

                outputs.append(torch.softmax(logits.detach().cpu(), dim=-1))
                labels.append(label)
                ids.extend(sample["id"])
        outputs = torch.cat(outputs)
        labels = torch.cat(labels)
        return agg_loss / (i + 1), labels, outputs, ids
