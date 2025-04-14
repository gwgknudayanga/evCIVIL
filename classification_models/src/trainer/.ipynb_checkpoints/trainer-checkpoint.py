import torch
import torch.nn as nn
import numpy as np

from typing import Tuple, List, Union, Type
import tqdm
from pathlib import Path
from ..datasets import Dataset
from torcheval.metrics.functional import multiclass_f1_score


class Trainer():
    def __init__(self, model: nn.Module, dataset: Union[None, Dataset]=None, saving_checkpoint: int = 100, path_to_save: Union[str, Path] = './data_train/', cfg: Union[None, dict] = None):
        self.cfg = cfg
        self.model = model
        self.dataset = dataset
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.device = None
        self.train_loader = None
        self.valid_loader = None
        self.epochs = 0
        self.checkpoint = saving_checkpoint
        self.path_to_save = path_to_save if isinstance(path_to_save, Path) else Path(path_to_save)
        if not self.path_to_save.exists():
            self.path_to_save.mkdir()
        self.train_losses = []
        self.valid_losses = []
        self.train_accs = []
        self.valid_accs = []
        self.f1_scores_train = []
        self.f1_scores_valid = []
        self.n_classes = 2 if dataset is None else dataset.n_classes  # so fukking ugly

    def _set_device(self, device: str) -> None:
        self.device = device
        self.model.to(self.device)
    
    def _set_criterion(self, criterion) -> None:
        self.criterion = criterion
    
    def _set_optimizer(self, optimizer) -> None:
        self.optimizer = optimizer
    
    def _set_scheduler(self, scheduler) -> None:
        self.scheduler = scheduler
    
    def _set_train_loader(self, ) -> None:
        self.train_loader = self.dataset.train_dataloader()
    
    def _set_valid_loader(self, ) -> None:
        self.valid_loader = self.dataset.valid_dataloader()

    def set_train_loader(self, train_dataloader: torch.utils.data.DataLoader) -> None:
        self.train_loader = train_dataloader
    
    def set_valid_loader(self, valid_dataloader: torch.utils.data.DataLoader) -> None:
        self.valid_loader = valid_dataloader
    
    def _set_epochs(self, epochs: int) -> None:
        self.epochs = epochs
    
    def init_trainer(self, device: str, criterion, optimizer, scheduler, epochs: int) -> None:
        self._set_device(device)
        self._set_criterion(criterion)
        self._set_optimizer(optimizer)
        self._set_scheduler(scheduler)
        if self.dataset is not None:
            self._set_train_loader()
            self._set_valid_loader()
        self._set_epochs(epochs)
    
    def train(self) -> None:
        for epoch in tqdm.tqdm(range(self.epochs)):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            predicted_labels = []
            true_labels = []
            for data in self.train_loader:
                if self.dataset is None:
                    _, inputs, labels = data
                    inputs, labels = inputs.to(self.device), labels.squeeze().type(torch.int64).to(self.device)
                else:
                    inputs, labels = data
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                if self.cfg.get("grad_clip", None):
                    nn.utils.clip_grad_value_(self.model.parameters(), self.cfg["grad_clip"])
                self.optimizer.step()

                # metrics
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                predicted_labels += predicted.cpu().numpy().tolist()
                true_labels += labels.cpu().numpy().tolist()
            # metrics
            self.f1_scores_train.append(
                multiclass_f1_score(
                    torch.tensor(predicted_labels),
                    torch.tensor(true_labels),
                    num_classes=self.n_classes,
                    average='macro'
                )
            )
            self.train_losses.append(running_loss/len(self.train_loader))
            self.train_accs.append(100 * correct / total)
            self._save_model(epoch)
            self.valid()
            self.scheduler.step(self.valid_accs[-1])

        self._save_data()
        # print('Finished Training')
    
    @torch.no_grad()
    def valid(self) -> None:
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        predicted_labels = []
        true_labels = []
        for data in self.valid_loader:
            if self.dataset is None:
                _, inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.squeeze().type(torch.int64).to(self.device)
            else:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.model(inputs)

            # metrics
            loss = self.criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            predicted_labels += predicted.cpu().numpy().tolist()
            true_labels += labels.cpu().numpy().tolist()
        # metrics
        self.valid_losses.append(running_loss/len(self.valid_loader))
        self.valid_accs.append(100 * correct / total)
        self.f1_scores_valid.append(
            multiclass_f1_score(
                torch.tensor(predicted_labels),
                torch.tensor(true_labels),
                num_classes=self.n_classes,
                average='macro'
            )
        )

    def _save_model(self, epoch: int) -> None:
        if (epoch + 1) % self.checkpoint == 0:
            torch.save(self.model.state_dict(), self.path_to_save / "checkpoints" / f"model_epoch_{epoch}.pt")
            self._save_data()
        if np.argmax(self.train_losses) == epoch:
            torch.save(self.model.state_dict(), self.path_to_save / "checkpoints" / f"best_train_model.pt")
    
    def load(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path))
    
    def _save_data(self) -> None:
        np.save(self.path_to_save / "metrics" / "train_losses.npy", np.array(self.train_losses))
        np.save(self.path_to_save / "metrics" / "valid_losses.npy", np.array(self.valid_losses))
        np.save(self.path_to_save / "metrics" / "train_accs.npy", np.array(self.train_accs))
        np.save(self.path_to_save / "metrics" / "valid_accs.npy", np.array(self.valid_accs))
        np.save(self.path_to_save / "metrics" / "f1_scores_train.npy", np.array(self.f1_scores_train))
        np.save(self.path_to_save / "metrics" / "f1_scores_valid.npy", np.array(self.f1_scores_valid))