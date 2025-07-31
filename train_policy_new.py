from pathlib import Path
import torch
from torch.utils.data import DataLoader

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

torch.manual_seed(42) 

def main():
    # path to save the trained model
    output_directory = Path("outputs/train/pusht_vn")
    output_directory.mkdir(parents=True, exist_ok=True)

    # use GPU for MacOS
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    training_steps = 5000
    log_freq = 1

    # Load Meta Data to eextract the camera keys and other details about data
    dataset_metadata = LeRobotDatasetMetadata("lerobot/pusht")
    # Then we grab all the image frames from the first camera:
    camera_key = dataset_metadata.camera_keys[0] # only use for now

    # instantiate the model
    # policy = VNPolicy(camera_key).to(device)
    # policy.train()

    # decide how to fetch the data for each batch

    delta_timestamps = {
        "observation.state": [-1.5, -1, -0.5, -0.20, -0.10, 0],  # (6, c); c is the dimension of the state space
        "action": [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],  # (16, c); c is the dimension of the action space
    }

    dataset = LeRobotDataset("lerobot/pusht", delta_timestamps=delta_timestamps)
    dataset.video_backend = "pyav" # to avoid TorchCodec issue
    dataloader = DataLoader(dataset, num_workers=4, batch_size=32, shuffle=True,
                            pin_memory=(device.type != "cpu"), drop_last=True)

    # define OPTIMIZER 
    optimizer = torch.optim.Adam(policy.parameters(), lr=LR)

    # training loop
    step = 0
    done = False
    while not done:
        for batch in dataloader:
            # print(f"{batch[camera_key].shape=}")  # (32, 4, c, h, w)
            # print(f"{batch['observation.state'].shape=}")  # (32, 6, c)
            # print(f"{batch['action'].shape=}")  # (32, 64, c)

            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            loss, _ = policy.forward(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if step % log_freq == 0:
                print(f"step: {step} loss: {loss.item():.3f}")
            step += 1
            if step >= training_steps:
                done = True
                break

    policy.save_pretrained(output_directory)


if __name__ == "__main__":
    main()
