import os
import random
from collections import OrderedDict
from typing import List, Optional

import einops
import numpy as np
import torch
import torch.nn as nn
import math

from torch.utils.data import random_split
import wandb


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


class eval_mode:
    def __init__(self, *models, no_grad=False):
        self.models = models
        self.no_grad = no_grad
        self.no_grad_context = torch.no_grad()

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)
        if self.no_grad:
            self.no_grad_context.__enter__()

    def __exit__(self, *args):
        if self.no_grad:
            self.no_grad_context.__exit__(*args)
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def freeze_module(module: nn.Module) -> nn.Module:
    for param in module.parameters():
        param.requires_grad = False
    module.eval()
    return module


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a, idx, axis=axis)


def transpose_batch_timestep(*args):
    return (einops.rearrange(arg, "b t ... -> t b ...") for arg in args)


class TrainWithLogger:
    def reset_log(self):
        self.log_components = OrderedDict()

    def log_append(self, log_key, length, loss_components):
        for key, value in loss_components.items():
            key_name = f"{log_key}/{key}"
            count, sum = self.log_components.get(key_name, (0, 0.0))
            self.log_components[key_name] = (
                count + length,
                sum + (length * value.detach().cpu().item()),
            )

    def flush_log(self, epoch, iterator=None):
        log_components = OrderedDict()
        iterator_log_component = OrderedDict()
        for key, value in self.log_components.items():
            count, sum = value
            to_log = sum / count
            log_components[key] = to_log
            # Set the iterator status
            log_key, name_key = key.split("/")
            iterator_log_name = f"{log_key[0]}{name_key[0]}".upper()
            iterator_log_component[iterator_log_name] = to_log
        postfix = ",".join(
            "{}:{:.2e}".format(key, iterator_log_component[key])
            for key in iterator_log_component.keys()
        )
        if iterator is not None:
            iterator.set_postfix_str(postfix)
        wandb.log(log_components, step=epoch)
        self.log_components = OrderedDict()


class SaveModule(nn.Module):
    def set_snapshot_path(self, path):
        self.snapshot_path = path
        print(f"Setting snapshot path to {self.snapshot_path}")

    def save_snapshot(self):
        os.makedirs(self.snapshot_path, exist_ok=True)
        torch.save(self.state_dict(), self.snapshot_path / "snapshot.pth")

    def load_snapshot(self):
        self.load_state_dict(torch.load(self.snapshot_path / "snapshot.pth"))


def split_datasets(dataset, train_fraction=0.95, random_seed=42):
    dataset_length = len(dataset) # number of samples, here each sample is a frame
    lengths = [
        int(train_fraction * dataset_length),
        dataset_length - int(train_fraction * dataset_length),
    ]
    train_set, val_set = random_split(
        dataset, lengths, generator=torch.Generator().manual_seed(random_seed)
    )
    return train_set, val_set

def split_dataset_episodewise_with_frame(full_dataset, train_fraction=0.95):
    '''
    Splits the full dataset based on the boundary frame based on training fraction, calcualtes the starting frame
    from the margin episode and splits the datasets.
    '''
    if full_dataset is not None:
        tot_eps = full_dataset.num_episodes

    boundary_eps = math.ceil(tot_eps * train_fraction) # after this episode, the test set will start
    boundary_fr = full_dataset.episode_data_index["from"][boundary_eps].item() #

    train_set = torch.utils.data.Subset(full_dataset, list(range(0, boundary_fr)))
    test_set = torch.utils.data.Subset(full_dataset, list(range(boundary_fr, full_dataset.num_frames)))

    return train_set, test_set