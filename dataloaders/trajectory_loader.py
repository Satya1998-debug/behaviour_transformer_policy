import torch
from torch.utils.data import TensorDataset, DataLoader
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from utils import (
    split_dataset_episodewise_with_frame,
)

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
    FPS = 10 # as per lerobot pushT dataset

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
        # get the frames from lerobot
        self.delta_timestamps = {
            "observation.state": [- t / self.fps for t in range(self.window_size)],  # (ws, 2); c is the dimension of the state space, window size 
            "action": [- t / self.fps for t in range(self.window_size)],  # (ws, 2); c is the dimension of the action space, window size
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
