# Modified Behavior Transformer (BeT) for PushT in LeRobot

This repository contains a **modified Behavior Transformer (BeT)** integrated with the [🤗 LeRobot](https://github.com/huggingface/lerobot) framework, trained and evaluated on the **PushT** robotic manipulation task.

The modification adapts BeT to the `lerobot/pusht` dataset and environment, using a transformer-based policy with **action discretization** and **offset correction** to handle multi-modal continuous actions.

---

## 📌 Installation & Environment Setup

LeRobot works with **Python 3.10+** and **PyTorch 2.2+**.  
We recommend using **conda**.

```bash
# Create and activate environment
conda create -y -n lerobot_bet python=3.10
conda activate lerobot_bet
```

Install ffmpeg
```bash
conda install ffmpeg -c conda-forge
```

Install LeRobot with PushT environment.
For simulations, 🤗 LeRobot comes with gymnasium environments that can be installed as extras:
- [pusht](https://github.com/huggingface/gym-pusht)

```bash
pip install -e ".[pusht]"
```

Install wandb for experiment tracking
```bash
pip install wandb
wandb login
```


## 📜 Acknowledgements
- [lerobot/pusht](https://github.com/huggingface/lerobot?tab=readme-ov-file):HuggingFace LeRobot for PushT dataset & environment
- [notmahi/BeT](https://github.com/notmahi/bet?tab=readme-ov-file): Multi-modal Behavior Transformer (BeT) architecture
- [karpathy/MinGPT](https://github.com/karpathy/minGPT): MinGPT implementation and hyperparameters.
- [facebookresearch/hydra](https://github.com/facebookresearch/hydra): Configuration managements.
- [psf/black](https://github.com/psf/black): Linting.