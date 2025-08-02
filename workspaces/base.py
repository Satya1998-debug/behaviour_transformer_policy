import logging
from collections import deque
from pathlib import Path

import einops
import gymnasium as gym
import hydra
import numpy as np
from numpy import linalg as LA
import torch
from models.action_ae.generators.base import GeneratorDataParallel
from models.latent_generators.latent_generator import LatentGeneratorDataParallel
import utils
import wandb
from torch.utils.data import DataLoader


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print("Saving to {}".format(self.work_dir))
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        utils.set_seed_everywhere(cfg.seed)
        self.helper_procs = []

        self.test_loader = None # for loading test dataset

        self.dataset = hydra.utils.call(  # calling dataset loader function using Hydra and passing all the necessary arguments
            cfg.env.dataset_fn,
            train_fraction=cfg.train_fraction,
            random_seed=cfg.seed,
            device=self.device,
        )


        self.train_set, self.test_set = self.dataset # get train and test sets
        self._setup_loaders() # setup data loaders for training and testing

        # self.env = gym.make(cfg.env.name)

        # Create the model
        self.action_ae = None
        self.obs_encoding_net = None
        self.state_prior = None
        if not self.cfg.lazy_init_models:
            self._init_action_ae()
            self._init_obs_encoding_net()
            self._init_state_prior()

        # wandb.init(dir=self.work_dir, project=cfg.project, config=cfg._content)
        self.epoch = 0
        self.load_snapshot() # load the snapshots

        # Set up history archival.
        self.window_size = cfg.window_size
        self.history = deque(maxlen=self.window_size)
        self.last_latents = None

    def _setup_loaders(self):

        self.test_loader = DataLoader(
            self.test_set,
            batch_size=self.cfg.test_batch_size,
            shuffle=False,
            num_workers=self.cfg.test_num_workers,
            pin_memory=True if self.device.type == "cuda" else False,  # in MPS, pin memory is not supported
            # drop_last=True
        )

    def _init_action_ae(self):
        if self.action_ae is None:  # possibly already initialized from snapshot
            self.action_ae = hydra.utils.instantiate(
                self.cfg.action_ae, _recursive_=False
            ).to(self.device)

    def _init_obs_encoding_net(self):
        if self.obs_encoding_net is None:  # possibly already initialized from snapshot
            self.obs_encoding_net = hydra.utils.instantiate(self.cfg.encoder)
            self.obs_encoding_net = self.obs_encoding_net.to(self.device)

    def _init_state_prior(self):
        if self.state_prior is None:  # possibly already initialized from snapshot
            self.state_prior = hydra.utils.instantiate(
                self.cfg.state_prior,
                latent_dim=self.action_ae.latent_dim,
                vocab_size=self.action_ae.num_latents,
            ).to(self.device)

            self.state_prior_optimizer = self.state_prior.get_optimizer(
                learning_rate=self.cfg.lr,
                weight_decay=self.cfg.weight_decay,
                betas=tuple(self.cfg.betas),
            )

    def _setup_plots(self):
        raise NotImplementedError

    def _setup_starting_state(self):
        raise NotImplementedError

    def _start_from_known(self):
        raise NotImplementedError

    def run_single_episode(self):
        obs_history = []
        action_history = []
        latent_history = []
        # obs = self.env.reset() # not needed, its just resets the flattened obs
        last_obs = obs
        if self.cfg.start_from_seen:
            obs = self._start_from_known()
        action, latents = self._get_action(obs, sample=True, keep_last_bins=False)
        done = False
        total_reward = 0
        obs_history.append(obs)
        action_history.append(action)
        latent_history.append(latents)
        for i in range(self.cfg.num_eval_steps):
            # if self.cfg.plot_interactions:
            #     self._plot_obs_and_actions(obs, action, done)
            if done:
                self._report_result_upon_completion()
                break
            # if self.cfg.enable_render:
            #     self.env.render(mode="human")
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if obs is None:
                obs = last_obs  # use cached observation in case of `None` observation
            else:
                last_obs = obs  # cache valid observation
            keep_last_bins = ((i + 1) % self.cfg.action_update_every) != 0
            action, latents = self._get_action(
                obs, sample=True, keep_last_bins=keep_last_bins
            )
            obs_history.append(obs)
            action_history.append(action)
            latent_history.append(latents)
        logging.info(f"Total reward: {total_reward}")
        logging.info(f"Final info: {info}")
        return total_reward, obs_history, action_history, latent_history, info

    def _report_result_upon_completion(self):
        pass

    def _plot_obs_and_actions(self, obs, chosen_action, done, all_actions=None):
        print(obs, chosen_action, done)
        raise NotImplementedError

    def _get_action(self, obs, sample=False, keep_last_bins=False):
        try: 
            with utils.eval_mode(
                self.action_ae, self.obs_encoding_net, self.state_prior, no_grad=True
            ):
                # obs = torch.from_numpy(obs).float().to(self.cfg.device).unsqueeze(0)
                enc_obs = self.obs_encoding_net(obs)  # encoded obs: # (batch, seq, dim=embed) (1 x 6 x 2)
                # enc_obs = einops.repeat(
                #     enc_obs, "obs -> batch obs", batch=self.cfg.test_batch_size
                # )
                # Now, add to history. This automatically handles the case where
                # the history is full.
                # self.history.append(enc_obs)
                if self.cfg.use_state_prior:
                    enc_obs_seq = einops.rearrange(enc_obs, "batch seq embed -> seq batch embed").to(self.cfg.device)  # type: ignore
                    # Sample latents from the prior
                    latents = self.state_prior.generate_latents(
                        enc_obs_seq,
                        torch.ones_like(enc_obs_seq).mean(dim=-1),
                    )
                
                    logits_to_save, offsets_to_save = None, None

                    offsets = None
                    if type(latents) is tuple:
                        latents, offsets = latents

                    if keep_last_bins and (self.last_latents is not None):
                        latents = self.last_latents
                    else:
                        self.last_latents = latents

                    # Take the final action latent
                    if self.cfg.enable_offsets:
                        action_latents = (latents[:, :, :], offsets[:, :, :])
                        # action_latents = (latents[:, -1:, :], offsets[:, -1:, :])
                    else:
                        action_latents = latents[:, -1:, :]
                else:
                    action_latents = self.action_ae.sample_latents(
                        num_latents=self.cfg.action_batch_size
                    )
                actions = self.action_ae.decode_actions(
                    latent_action_batch=action_latents,
                    input_rep_batch=enc_obs,
                )
                actions = actions.cpu().numpy()
                if sample:
                    sampled_action = np.random.randint(len(actions))
                    actions = actions[sampled_action]
                    # (seq==1, action_dim), since batch dim reduced by sampling
                    actions = einops.rearrange(actions, "seq action_dim -> 1 seq action_dim")
                else:
                    # (batch, seq==1, action_dim)
                    actions = einops.rearrange(
                        actions, "batch seq action_dim -> batch seq action_dim"
                    )
                return actions, (logits_to_save, offsets_to_save, action_latents)
        except Exception as e:
            logging.error(f"Error in _get_action: {e}")
            raise e

    def run(self):
        rewards = []
        infos = []
        if self.cfg.lazy_init_models: # initialize models
            self._init_action_ae()
            self._init_obs_encoding_net()
            self._init_state_prior()
        for i in range(self.cfg.num_eval_eps):
            reward, obses, actions, latents, info = self.run_single_episode()
            rewards.append(reward)
            infos.append(info)
            torch.save(actions, Path.cwd() / f"actions_{i}.pth")
            torch.save(latents, Path.cwd() / f"latents_{i}.pth")
            
        self.env.close()
        logging.info(rewards)
        logging.info(infos)
        return rewards, infos

    @property
    def snapshot(self):
        return Path(self.cfg.load_dir or self.work_dir) / "snapshot.pt"

    def load_snapshot(self):
        keys_to_load = ["action_ae", "obs_encoding_net", "state_prior"]
        with self.snapshot.open("rb") as f:
            payload = torch.load(f, map_location=self.device, weights_only=False)
        loaded_keys = []
        for k, v in payload.items():
            if k in keys_to_load:
                loaded_keys.append(k)
                self.__dict__[k] = v.to(self.cfg.device)

        if len(loaded_keys) != len(keys_to_load):
            raise ValueError(
                "Snapshot does not contain the following keys: "
                f"{set(keys_to_load) - set(loaded_keys)}"
            )
        
    def normalized_error(self, pred, true, eps=1e-8):
        error = torch.norm(pred - true, dim=-1)
        norm_true = torch.norm(true, dim=-1)
        return error / (norm_true + eps)
    
    def accuracy_from_normalized_error(self, norm_err, threshold=0.1):
        correct = (norm_err < threshold).float()
        accuracy = correct.mean().item()
        return accuracy
        
    def evaluate_action_prediction(self):
        try:
            tot_acc = 0.0
            count = 1

            # self.history.clear()
            num_batches = len(self.test_loader)

            for batch in self.test_loader:
                observations, true_action = batch['observation.state'], batch['action']
                pred_action, _ = self._get_action(observations, sample=True)

                # Flatten if needed
                pred_action_fl = np.asarray(pred_action).flatten()
                true_action_fl = np.asarray(true_action).flatten()

                # error = np.norm((pred_action - true_action) ** 2)
                norm_err = self.normalized_error(torch.from_numpy(pred_action_fl), torch.from_numpy(true_action_fl))
                accuracy = self.accuracy_from_normalized_error(norm_err)
                # print(f'Accuracy for batch-{count}:', accuracy)
                count += 1
                tot_acc += accuracy

            avg_acc = tot_acc / count
            print(f"\nAverage accuracy over {count} batches: {avg_acc:.4f}")
            return avg_acc * 100
        except Exception as e:
            logging.error(f"Error in evaluate_action_prediction: {e}")
            raise e
