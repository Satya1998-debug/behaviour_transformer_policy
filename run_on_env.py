import hydra
import joblib
import numpy as np
from workspaces.base import Workspace


@hydra.main(config_path="configs", config_name="eval_pusht")
def main(cfg):
    # Needs _recursive_: False since we have more objects within that we are instantiating
    # without using nested instantiation from hydra
    # workspace = hydra.utils.instantiate(cfg.env.workspace, cfg=cfg, _recursive_=False)
    workspace = Workspace(cfg)
    rewards, infos = workspace.run()
    print(rewards)
    print(infos)
    print(f"Average reward: {np.mean(rewards)}")
    print(f"Std: {np.std(rewards)}")


if __name__ == "__main__":
    main()
