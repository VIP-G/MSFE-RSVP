"""SimpleDataSet dataloader used by the MSFE-RSVP training loop."""

import copy
import importlib
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler


DATASET_MODULES = {"SimpleDataSet": "tools.data.simple_dataset"}


def seed_data_worker(worker_id: int) -> None:
    """Synchronize Python and NumPy seeds in each data worker."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_dataloader(config, mode, logger, seed=None, epoch=1, task="rec"):
    """Build the final SimpleDataSet DataLoader."""
    if task != "rec":
        raise ValueError("The MSFE-RSVP repository only supports rec tasks.")

    config = copy.deepcopy(config)
    mode = mode.capitalize()
    dataset_config = config[mode]["dataset"]
    module_name = dataset_config["name"]
    if module_name not in DATASET_MODULES:
        raise ValueError(
            f"Unsupported dataset: {module_name}; only SimpleDataSet is "
            "available"
        )

    loader_randomness = {}
    sampling_seed = dataset_config.get("sampling_seed")
    if mode == "Train" and sampling_seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(sampling_seed) + int(epoch) - 1)
        loader_randomness = {
            "generator": generator,
            "worker_init_fn": seed_data_worker,
        }

    dataset_module = importlib.import_module(DATASET_MODULES[module_name])
    dataset_class = getattr(dataset_module, module_name)
    dataset = dataset_class(config, mode, logger, seed, epoch=epoch, task=task)

    loader_config = config[mode]["loader"]
    sampler = None
    if config["Global"].get("distributed") and mode == "Train":
        sampler = DistributedSampler(
            dataset=dataset, shuffle=loader_config["shuffle"]
        )

    collate_fn = None
    if hasattr(dataset, "collate_fn"):
        collate_fn = dataset.collate_fn
    elif "collate_fn" in loader_config:
        raise ValueError("custom collate_fn is not supported in this repository")

    data_loader = DataLoader(
        dataset=dataset,
        sampler=sampler,
        num_workers=loader_config["num_workers"],
        pin_memory=loader_config.get("pin_memory", False),
        collate_fn=collate_fn,
        batch_size=loader_config["batch_size_per_card"],
        drop_last=loader_config["drop_last"],
        shuffle=loader_config["shuffle"] if sampler is None else False,
        **loader_randomness,
    )
    if len(data_loader) == 0:
        logger.error(
            "No images in the dataloader. Check label files and batch size."
        )
        sys.exit()
    return data_loader
