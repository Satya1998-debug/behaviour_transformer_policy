import logging
from collections import OrderedDict
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader

from models.action_ae.generators.base import GeneratorDataParallel
from models.latent_generators.latent_generator import LatentGeneratorDataParallel
from omegaconf import OmegaConf

import utils
import wandb


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print("Saving to {}".format(self.work_dir))
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        utils.set_seed_everywhere(cfg.seed)

        self.dataset = hydra.utils.call(  # calling dataset loader function using Hydra and passing all the necessary arguments
            cfg.env.dataset_fn,
            train_fraction=cfg.train_fraction,
            random_seed=cfg.seed,
            device=self.device,
        )


        self.train_set, self.test_set = self.dataset # get train and test sets
        self._setup_loaders() # setup data loaders for training and testing

        # Create the model
        self.action_ae = None
        self.obs_encoding_net = None
        self.state_prior = None  # usually the transformer model
        if not self.cfg.lazy_init_models:
            self._init_action_ae()
            self._init_obs_encoding_net()
            self._init_state_prior()

        self.log_components = OrderedDict()
        self.epoch = self.prior_epoch = 0

        self.save_training_latents = False
        self._training_latents = []

        self.wandb_run = wandb.init(
            dir=str(self.work_dir),
            project=cfg.project,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        wandb.config.update(
            {
                "save_path": self.work_dir,
            }
        )

    def _init_action_ae(self):
        if self.action_ae is None:  # possibly already initialized from snapshot
            self.action_ae = hydra.utils.instantiate(  # KMeansDiscretizer object from models/action_ae/discretizers/k_means.py is returned
                self.cfg.action_ae, _recursive_=False
            ).to(self.device)

            if self.cfg.data_parallel:  # not used
                self.action_ae = GeneratorDataParallel(self.action_ae)

    def _init_obs_encoding_net(self):
        if self.obs_encoding_net is None:  # possibly already initialized from snapshot
            self.obs_encoding_net = hydra.utils.instantiate(self.cfg.encoder)  # nn.Identity object is returned
            self.obs_encoding_net = self.obs_encoding_net.to(self.device)

            if self.cfg.data_parallel:  # not used
                self.obs_encoding_net = torch.nn.DataParallel(self.obs_encoding_net)

    def _init_state_prior(self):
        if self.state_prior is None:  # possibly already initialized from snapshot
            self.state_prior = hydra.utils.instantiate(  # models.latent_generators.mingpt.MinGPT object is returned
                self.cfg.state_prior,
                latent_dim=self.action_ae.latent_dim,  # latent_dim = 1
                vocab_size=self.action_ae.num_latents, # number of bins in the discretizer
            ).to(self.device)

            if self.cfg.data_parallel:  # not used
                self.state_prior = LatentGeneratorDataParallel(self.state_prior)

            self.state_prior_optimizer = self.state_prior.get_optimizer( # Adam optimizer for the state prior model
                learning_rate=self.cfg.lr,
                weight_decay=self.cfg.weight_decay,
                betas=tuple(self.cfg.betas),
            )

    def _setup_loaders(self):
        self.train_loader = DataLoader(
            self.train_set,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )

        self.test_loader = DataLoader(
            self.test_set,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )

        # self.latent_collection_loader = DataLoader(
        #     self.train_set,
        #     batch_size=self.cfg.batch_size,
        #     shuffle=False,
        #     num_workers=self.cfg.num_workers,
        #     pin_memory=True,
        # )

    def train_prior(self):
        self.state_prior.train()
        with utils.eval_mode(self.obs_encoding_net, self.action_ae):
            pbar = tqdm.tqdm(
                self.train_loader, desc=f"Training prior epoch {self.prior_epoch}"
            )
            for batch in pbar:
                observations, action = batch['observation.state'], batch['action']
                self.state_prior_optimizer.zero_grad(set_to_none=True)
                obs, act = observations.to(self.device), action.to(self.device)
                enc_obs = self.obs_encoding_net(obs)  # does nothing, only identity operation
                latent = self.action_ae.encode_into_latent(act, enc_obs)
                _, loss, loss_components = self.state_prior.get_latent_and_loss(
                    obs_rep=enc_obs,
                    target_latents=latent,
                    return_loss_components=True,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.state_prior.parameters(), self.cfg.grad_norm_clip
                )
                self.state_prior_optimizer.step()
                self.log_append("prior_train", len(observations), loss_components)

    def eval_prior(self):
        with utils.eval_mode(
            self.obs_encoding_net, self.action_ae, self.state_prior, no_grad=True
        ):
            for batch in self.test_loader:
                obs, act = batch['observation.state'].to(self.device), batch['action'].to(self.device)
                enc_obs = self.obs_encoding_net(obs)
                latent = self.action_ae.encode_into_latent(act, enc_obs)
                _, loss, loss_components = self.state_prior.get_latent_and_loss(
                    obs_rep=enc_obs,
                    target_latents=latent,
                    return_loss_components=True,
                )
                self.log_append("prior_eval", len(obs), loss_components)

    def run(self):
        snapshot = self.snapshot
        if snapshot.exists():
            print(f"Resuming: {snapshot}")
            self.load_snapshot()

        if self.cfg.lazy_init_models:
            self._init_obs_encoding_net() # initialize observation encoding
            self._init_action_ae() # initialize action autoencoder

        self.action_ae.fit_model(  # train the action autoencoder
            self.train_loader,
            self.test_loader,
            self.obs_encoding_net,
        )
        # if self.cfg.save_latents: # save the latents after training the autoencoder, but now its False
        #     self.save_latents()

        # Train the action prior model.
        if self.cfg.lazy_init_models:
            self._init_state_prior()

        self.state_prior_iterator = tqdm.trange(
            self.prior_epoch, self.cfg.num_prior_epochs
        )
        self.state_prior_iterator.set_description("Training prior: ")

        # Reset the log.
        self.log_components = OrderedDict()

        for epoch in self.state_prior_iterator:
            self.prior_epoch = epoch

            # train the Prior model
            self.train_prior()

            # evaluate the Prior model every `eval_prior_every` epochs
            if ((self.prior_epoch + 1) % self.cfg.eval_prior_every) == 0:
                self.eval_prior() # evaluate the Prior model

            self.flush_log(epoch=epoch + self.epoch, iterator=self.state_prior_iterator)
            self.prior_epoch += 1

            # save the snapshot every `save_prior_every` epochs
            if ((self.prior_epoch + 1) % self.cfg.save_prior_every) == 0:
                self.save_snapshot()

        # expose DataParallel module class name for wandb tags
        tag_func = (
            lambda m: m.module.__class__.__name__
            if self.cfg.data_parallel
            else m.__class__.__name__
        )
        tags = tuple(
            map(tag_func, [self.obs_encoding_net, self.action_ae, self.state_prior])
        )
        self.wandb_run.tags += tags

    @property
    def snapshot(self):
        return self.work_dir / "snapshot.pt"

    def save_snapshot(self):
        self._keys_to_save = [
            "action_ae",
            "obs_encoding_net",
            "epoch",
            "prior_epoch",
            "state_prior",
        ]
        payload = {k: self.__dict__[k] for k in self._keys_to_save}
        with self.snapshot.open("wb") as f:
            torch.save(payload, f)

    def save_latents(self):
        total_mse_loss = 0
        with utils.eval_mode(self.action_ae, self.obs_encoding_net, no_grad=True):
            for observations, action, mask in self.latent_collection_loader:
                obs, act = observations.to(self.device), action.to(self.device)
                enc_obs = self.obs_encoding_net(obs)
                latent = self.action_ae.encode_into_latent(act, enc_obs)
                reconstructed_action = self.action_ae.decode_actions(
                    latent,
                    enc_obs,
                )
                total_mse_loss += F.mse_loss(act, reconstructed_action, reduction="sum")
                if type(latent) == tuple:
                    # serialize into tensor; assumes last dim is latent dim
                    detached_latents = tuple(x.detach() for x in latent)
                    self._training_latents.append(torch.cat(detached_latents, dim=-1))
                else:
                    self._training_latents.append(latent.detach())
        self._training_latents_tensor = torch.cat(self._training_latents, dim=0)
        logging.info(f"Total MSE reconstruction loss: {total_mse_loss}")
        logging.info(
            f"Average MSE reconstruction loss: {total_mse_loss / len(self._training_latents_tensor)}"
        )
        torch.save(self._training_latents_tensor, self.work_dir / "latents.pt")

    def load_snapshot(self):
        with self.snapshot.open("rb") as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v
        not_in_payload = set(self._keys_to_save) - set(payload.keys())
        if len(not_in_payload):
            logging.warning("Keys not found in snapshot: %s", not_in_payload)

    def log_append(self, log_key, length, loss_components):
        for key, value in loss_components.items():
            key_name = f"{log_key}/{key}"
            count, sum = self.log_components.get(key_name, (0, 0.0))
            self.log_components[key_name] = (
                count + length,
                sum + (length * value.detach().cpu().item()),
            )

    def flush_log(self, epoch, iterator):
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
        iterator.set_postfix_str(postfix)
        wandb.log(log_components, step=epoch)
        self.log_components = OrderedDict()


@hydra.main(config_path="configs", config_name="config")
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
