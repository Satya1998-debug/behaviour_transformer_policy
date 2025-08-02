import logging
import einops
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, Dataset, DataLoader
from pathlib import Path
import numpy as np
import lerobot
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from utils import (
    shuffle_along_axis,
    transpose_batch_timestep,
    split_datasets,
    split_dataset_episodewise_with_frame,
    eval_mode,
)
from typing import Union, Callable, Optional
from tqdm import tqdm


class PushTrajectoryDataset(TensorDataset):
    def __init__(
        self,
        data_directory: os.PathLike,
        device="cpu",
    ):
        self.device = device
        self.data_directory = Path(data_directory)
        logging.info("Multimodal loading: started")
        self.observations = np.load(
            self.data_directory / "multimodal_push_observations.npy"
        )
        self.actions = np.load(self.data_directory / "multimodal_push_actions.npy")
        self.masks = np.load(self.data_directory / "multimodal_push_masks.npy")
        self.observations = torch.from_numpy(self.observations).to(device).float()
        self.actions = torch.from_numpy(self.actions).to(device).float()
        self.masks = torch.from_numpy(self.masks).to(device).float()
        logging.info("Multimodal loading: done")
        # The current values are in shape N x T x Dim, so all is good in the world.
        super().__init__(
            self.observations,
            self.actions,
            self.masks,
        )

    def get_seq_length(self, idx):
        return int(self.masks[idx].sum().item())

    def get_all_actions(self):
        result = []
        # mask out invalid actions
        for i in range(len(self.masks)):
            T = int(self.masks[i].sum().item())
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)


class GridTrajectoryDataset(TensorDataset):
    def __init__(
        self,
        grid_size=5,
        device="cpu",
        num_samples=1_000_000,
        top_prob=0.4,
        noise_scale=0.05,
        random_seed=42,
        scale_factor=1.0,
    ):
        rng = np.random.default_rng(random_seed)
        total_grid_size = grid_size * 2
        top_length = int(total_grid_size * top_prob)
        side_length = total_grid_size - top_length

        all_up_actions = np.concatenate(
            [np.ones((num_samples, top_length)), np.zeros((num_samples, side_length))],
            axis=-1,
        ).astype(float)
        all_up_actions = shuffle_along_axis(all_up_actions, axis=-1)
        all_side_actions = 1.0 - all_up_actions
        all_actions = np.stack([all_up_actions, all_side_actions], axis=-1)
        all_observations = np.cumsum(all_actions, axis=1)  # [N, T, 2]
        all_actions += rng.normal(scale=noise_scale, size=all_actions.shape)
        all_observations += rng.normal(scale=noise_scale, size=all_observations.shape)

        # Scale the actions to be between 0 and scale_factor
        all_observations, all_actions = (
            scale_factor * all_observations,
            scale_factor * all_actions,
        )
        # All cells are valid
        mask = np.ones(all_observations.shape[:-1])
        self.mask = mask

        super().__init__(
            torch.from_numpy(all_observations).to(device).float(),
            torch.from_numpy(all_actions).to(device).float(),
            torch.from_numpy(mask).to(device).float(),
        )

    def get_seq_length(self, idx) -> int:
        return int(self.mask[idx].sum().item())


class TrajectorySlicerDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        window: int,
        transform: Optional[Callable] = None,
    ):
        """
        Slice a trajectory dataset into unique (but overlapping) sequences of length `window`.

        dataset: a trajectory dataset that satisfies:
            dataset.get_seq_length(i) is implemented to return the length of sequence i
            dataset[i] = (observations, actions, mask)
            observations: Tensor[T, ...]
            actions: Tensor[T, ...]
            mask: Tensor[T]
                0: invalid
                1: valid
        window: int
            number of timesteps to include in each slice
        returns: a dataset of sequences of length `window`
        """
        self.dataset = dataset
        self.window = window
        self.transform = transform
        self.slices = []
        min_seq_length = np.inf
        for i in range(len(self.dataset)):  # type: ignore
            T = self._get_seq_length(i)  # avoid reading actual seq (slow)
            min_seq_length = min(T, min_seq_length)
            if T - window < 0:
                print(f"Ignored short sequence #{i}: len={T}, window={window}")
            else:
                self.slices += [
                    (i, start, start + window) for start in range(T - window)
                ]  # slice indices follow convention [start, end)

            if min_seq_length < window:
                print(
                    f"Ignored short sequences. To include all, set window <= {min_seq_length}."
                )

    def _get_seq_length(self, idx: int) -> int:
        # Adding this convenience method to avoid reading the actual sequence
        # We retrieve the length in trajectory slicer just so we can use subsetting
        # and shuffling before we pass a dataset into TrajectorySlicerDataset
        return self.dataset.get_seq_length(idx)

    def _get_all_actions(self) -> torch.Tensor:
        return self.dataset.get_all_actions()

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        i, start, end = self.slices[idx]
        values = tuple(
            x[start:end] for x in self.dataset[i]
        )  # (observations, actions, mask)
        # optionally apply transform
        if self.transform is not None:
            values = self.transform(values)
        return values


class TrajectorySlicerSubset(TrajectorySlicerDataset):
    def _get_seq_length(self, idx: int) -> int:
        # self.dataset is a torch.dataset.Subset, so we need to use the parent dataset
        # to extract the true seq length.
        subset = self.dataset
        return subset.dataset.get_seq_length(subset.indices[idx])  # type: ignore

    def _get_all_actions(self) -> torch.Tensor:
        return self.dataset.dataset.get_all_actions()


class TrajectoryRepDataset(Dataset):
    def __init__(
        self,
        trajectory_dataset: Dataset,
        encoder: nn.Module,
        preprocess: Callable[[torch.Tensor], torch.Tensor] = None,
        postprocess: Callable[[torch.Tensor], torch.Tensor] = None,
        device: Union[torch.device, str] = "cuda",
        batch_size: Optional[int] = 128,
    ):
        """
        Given a trajectory dataset, encode its states into representations.
        Inputs:
            trajectory_dataset: a trajectory dataset that satisfies:
                dataset[i] = (observations, actions, mask)
                observations: Tensor[T, ...]
                actions: Tensor[T, ...]
                masks: Tensor[T]
                    0: invalid
                    1: valid
            encoder: a module that accepts observations and returns a representation
            device: encoder will be run on this device
            batch_size: if not None, will batch frames into batches of this size (to avoid OOM)
        """
        self.device = device
        encoder = encoder.to(device)  # not saving encoder to lower VRAM usage
        self.obs = []
        self.actions = []
        self.masks = []
        self.postprocess = postprocess
        with eval_mode(encoder, no_grad=True):
            for i in tqdm(range(len(trajectory_dataset))):
                obs, act, mask = trajectory_dataset[i]
                if preprocess is not None:
                    obs = preprocess(obs)
                if batch_size is not None:
                    obs_enc = []
                    for t in range(0, obs.shape[0], batch_size):
                        batch = obs[t : t + batch_size].to(self.device)
                        obs_enc.append(encoder(batch).cpu())
                    obs_enc = torch.cat(obs_enc, dim=0)
                else:
                    obs_enc = encoder(obs.to(self.device)).cpu()
                self.obs.append(obs_enc)
                self.actions.append(act)
                self.masks.append(mask)
        del encoder
        torch.cuda.empty_cache()

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, idx):
        obs = self.obs[idx]
        if self.postprocess is not None:
            obs = self.postprocess(obs)
        return (obs, self.actions[idx], self.masks[idx])

    def get_seq_length(self, idx):
        return int(self.masks[idx].sum().item())

    def get_all_actions(self):
        return torch.cat(self.actions, dim=0)


def get_push_train_val(
    data_directory,
    train_fraction=0.9,
    random_seed=42,
    device="cpu",
    window_size=10,
):
    push_trajectories = PushTrajectoryDataset(data_directory)
    train_set, val_set = split_datasets(
        push_trajectories,
        train_fraction=train_fraction,
        random_seed=random_seed,
    )
    # Convert to trajectory slices.
    train_trajectories = TrajectorySlicerSubset(train_set, window=window_size)
    val_trajectories = TrajectorySlicerSubset(val_set, window=window_size)
    return train_trajectories, val_trajectories

def get_pusht_train_val(dataset_path,
    train_fraction=0.9,
    random_seed=42,
    device="cpu",
    window_size=6):
    """
    Get the train and validation datasets for the PushT environment.
    Returns:
        train_set: Training dataset with trajectory slices.
        val_set: Validation dataset with trajectory slices.
    """
    FPS = 10

    push_dataset = PushTDataset(train_fraction=train_fraction, 
                                dataset_path=dataset_path,
                                random_seed=random_seed, 
                                device=device, 
                                window_size=window_size,
                                fps=FPS)
    push_dataset.setup()
    # Get the train and test datasets
    # This will return the train and test datasets as per the split defined in PushTDataset
    train_dataset, test_dataset = push_dataset.get_dataset()
    return train_dataset, test_dataset

class PushTDataset(TensorDataset):
    def __init__(self, dataset_path: str = "lerobot/pusht", 
                 train_fraction: float = 0.9, 
                 random_seed: int = 42, 
                 window_size: int = 6,
                 batch_size: int = 32,
                 num_workers: int = 8, 
                 device: torch.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu"),
                 fps: int = 10,
    ):
        self.dataset_path = dataset_path
        self.train_fraction = train_fraction
        self.random_seed = random_seed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.window_size = window_size
        self.fps = fps

        self.dataset = None
        self.train_set = None
        self.test_set = None

    def setup(self):
        # sets the delta timestamps for the observations and actions,(window_size - 1) samples from past and 1 current sample
        self.delta_timestamps = {
            "observation.state": [- t / self.fps for t in range(self.window_size)],  # (6, c); c is the dimension of the state space, window size is 6
            "action": [- t / self.fps for t in range(self.window_size)],  # (6, c); c is the dimension of the action space, window size is 6
        }
        # Load the full dataset
        self.dataset = LeRobotDataset(self.dataset_path, delta_timestamps=self.delta_timestamps)
        self.dataset.video_backend = "pyav"  # Avoid TorchCodec issues

        # split the datasets based on episodes, some episodes are used for training and some for validation
        self.train_set, self.test_set = split_dataset_episodewise_with_frame(self.dataset, train_fraction=self.train_fraction)

    # TODO: NOT used for now
    def get_dataloader(self, split: str = "train", shuffle: bool = True) -> DataLoader:
        if split == "train":
            dataset = self.train_set
        elif split in ["val", "test"]:
            dataset = self.test_set
            shuffle = False
        else:
            raise ValueError("split must be 'train', 'val', or 'test'")

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=(self.device.type != "cpu"),
            drop_last=True,
        )

    def get_dataset(self):
        return self.train_set, self.test_set
