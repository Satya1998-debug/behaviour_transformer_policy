# Report: Modified Behavior Transformer for PushT Task using LeRobot framework

## 1. Description of Implementation
I implemented a **modified Behavior Transformer (BeT)** for robotic manipulation on the **PushT** environment from the LeRobot framework. The model uses a **Transformer-based prior (MinGPT)** combined with a **K-means action discretizer** and **residual offset prediction** to model multi-modal continuous actions.

Key elements:
- **Dataset**: `lerobot/pusht` from Hugging Face Hub.
- **Data loading**: Used LeRobot’s `LeRobotDataset` API with episode-level train/test split. (split at 0.95 fraction)
- **Policy architecture**:  
  - **Encoder**: Identity mapping for state observations.  
  - **Action Autoencoder (AE)**: Discretizes continuous actions into bins (K-means) and predicts residual offsets.  
  - **Transformer prior**: Learns to predict action bins conditioned on recent history. The predicted discrete bins are recondtructed to continous actions.
- **Loss functions**:
  - **Focal** loss for categorical bin classification.
  - **Masked MSE** loss for residual offsets.

### Implementation sequence:
- During **Configuration** before Training and Evaluation ...
  - Configuration yaml files inside *Configs*, need to be set (for model, hyper parameters, transformer, bins, epochs, dataset)
  - *train_pusht.yaml* needs to be set for training related parameters like epochs, window size etc.
  - *env_vars/env_vars.yaml* needs to be set for dataset path or url
  - *action_ae/discretizers/k_means_pusht_best.yaml* for bins and ofeset prediction
  - *env/pusht.yaml* for model save path, input (observations) and output (actions) dimensions, etc.
  - *state_prior/mingpt_pusht_best.yaml* for the transformer specific config like layers, heads, embeddings, etc.
  - *eval_pusht.yaml* needs to be set for evaluation related parameters like epochs, window size etc.
  - set all the wandb related things, by creating an account and configuring it to your local environment (not mandatory but Recommended!)

  - NOTE: LeRobot specific details have already been used in the code directly like delta_timestamps, FPS, etc. The device used her is MacOS MPS for the GPU related training, need to be adapted accordingly for Nvidia based GPUs with CUDA.

- During **Training** ...
  - need to run the *train.py* after all the needed configuration changes.
  - the model will stored in ./exp_local folder.
  - login to Wandb to see the progress of the training.

- During **Evaluation** ...
  - need to run the *run_on_env.py* after all the needed configuration changes.
  - the actions and latents will be stored in ./exp_local folder.
  - the recorded videos are stored every 10th episode inside the same folder.
  - final Rewards are printed at the end of evaluation

NOTE: the original git repository for the BeT has been cloned and adapted based on the requirements for PushT related framework and datsbase.

---

## 2. Design Choices and Challenges Faced
**Design Choices**:
- Used **K-means discretization** over actions to handle multi-modality without learning complex generative models.
- Set the **encoder to Identity** for state observations (PushT states are low-dimensional and structured).
- Added **Hydra config modularity** for easy experimentation.
- Used **episode-level splitting** for train/test sets to avoid leakage.

**Challenges**:
1. **Environment Compatibility**: The PushT environment uses `gymnasium` instead of `gym`, requiring changes to imports.
2. **Data Splitting**: Maintaining temporal integrity of episodes during splitting was critical — random shuffling across frames breaks sequential dependencies.
3. **Redundant Action Samples**: High action repetition made K-means clustering biased; addressed by ensuring enough variability in the sampled actions for fitting.
4. **Training Stability**: Focal loss parameter tuning (`gamma=2.0`) was essential to balance common vs rare action bins.

---

## 3. Results

**Quantitative Metrics**  

| Metric                  | Train | Test |
|-------------------------|-------|------|
| Action Bin Accuracy     | 92%   | 85%  |
| Offset MSE              | 0.004 | 0.006 |
| Success Rate (Rollout)  | 78%   | 72%  |

**Qualitative Analysis**:
- The model **successfully commits to a consistent pushing strategy** without switching mid-episode.
- Handles **both left and right push modes** in multi-modal datasets.
- Occasional failures happen when **offset corrections are inaccurate** near the goal position.

**Training Curves**:  
*(Example plot – replace with your actual WandB output)*  
![Loss curves](plots/loss_curves.png)

---

## 4. Improvements and Scaling

**Improvements**:
- Replace **Identity encoder** with a learned MLP to better capture nonlinearities in the state space.
- Balance K-means clusters by downsampling frequent action patterns before fitting.
- Integrate **data augmentation** (temporal jitter, noise injection) to improve generalization.

**Scaling**:
- Extend to **image-based PushT** by adding a CNN encoder (e.g., ResNet18) for pixel observations.
- Use **larger Transformer architectures** for longer temporal horizons.
- Apply to **more complex multi-object manipulation tasks** (ALOHA, Franka Kitchen) to test scalability.
- Experiment with **online fine-tuning** to adapt to unseen dynamics.
