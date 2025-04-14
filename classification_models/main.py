import torch
import torch.nn as nn
import numpy as np

from src.trainer.trainer import Trainer
from src import models
from src import datasets

from pathlib import Path
import random
import hydra
import datetime
import yaml
from omegaconf import OmegaConf


def worker_seed_set(worker_id):
    # See for details of numpy:
    # https://github.com/pytorch/pytorch/issues/5059#issuecomment-817392562
    # See for details of random:
    # https://pytorch.org/docs/stable/notes/randomness.html#dataloader

    # NumPy
    uint64_seed = torch.initial_seed()
    ss = np.random.SeedSequence([uint64_seed])
    np.random.seed(ss.generate_state(4))

    # random
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)

def init_path(cfg):
    path_to_save = Path(cfg["path_to_save"]) / datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f")
    path_to_save.mkdir(parents=True, exist_ok=True)
    (path_to_save / "checkpoints").mkdir(parents=True, exist_ok=True)
    (path_to_save / "metrics").mkdir(parents=True, exist_ok=True)
    
    cfg = OmegaConf.to_container(cfg, resolve=True)
    with open(path_to_save / "cfg.yaml", 'w') as f:
        yaml.dump(cfg, f)
    return path_to_save


@hydra.main(version_base=None, config_name="cfg", config_path="./")
def main_event(cfg):
    # create folders and save configurations
    path_to_save = init_path(cfg)

    # load model and dataset
    train_dataset = getattr(datasets, cfg["dataset"])(mode="train", **cfg["dataset_train_params"])
    valid_dataset = getattr(datasets, cfg["dataset"])(mode="val", **cfg["dataset_test_params"])

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=10,
        pin_memory=True,
        worker_init_fn=worker_seed_set,
        drop_last=True,
    )
    valid_dataloder = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=10,
        pin_memory=True,
    )

    model = getattr(models, cfg["model"])(
        img_channels=cfg["img_channels"],
        num_classes=train_dataset.n_classes,
    )

    # create trainer
    trainer = Trainer(
        model,
        None,
        saving_checkpoint=cfg["saving_checkpoint"],
        path_to_save=path_to_save,
        cfg=cfg
    )
    criterion = getattr(nn, cfg["criterion"])()
    optimizer = getattr(torch.optim, cfg["optimizer"])(model.parameters(), **cfg["optimizer_params"])
    scheduler = getattr(torch.optim.lr_scheduler, cfg["scheduler"])(
        optimizer,
        **cfg["scheduler_params"],
    )
    trainer.init_trainer(
        device=cfg["device"],
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=cfg["epochs"],
    )
    trainer.set_train_loader(train_dataloader)
    trainer.set_valid_loader(valid_dataloder)

    # training
    trainer.train()


@hydra.main(version_base=None, config_name="cfg_others", config_path="./")
def main_others(cfg):
    # create folders and save configurations
    path_to_save = init_path(cfg)

    # load model and dataset
    dataset = getattr(datasets, cfg["dataset"])(**cfg["dataset_params"])
    model = getattr(models, cfg["model"])(
        img_channels=cfg["img_channels"],
        num_classes=dataset.n_classes,
    )
    
    # create trainer
    trainer = Trainer(
        model,
        dataset,
        saving_checkpoint=cfg["saving_checkpoint"],
        path_to_save=path_to_save,
        cfg=cfg
    )
    criterion = getattr(nn, cfg["criterion"])()
    optimizer = getattr(torch.optim, cfg["optimizer"])(model.parameters(), **cfg["optimizer_params"])
    scheduler = getattr(torch.optim.lr_scheduler, cfg["scheduler"])(
        optimizer,
        **cfg["scheduler_params"],
    )
    trainer.init_trainer(
        device=cfg["device"],
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=cfg["epochs"],
    )

    # training
    trainer.train()


if __name__ == '__main__':
    main_event()
    # main_others()

    