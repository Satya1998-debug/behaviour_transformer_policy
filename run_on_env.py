import hydra
import joblib
import numpy as np
from workspaces.base import Workspace


@hydra.main(config_path="configs", config_name="env_config")
def main(cfg):
    # Needs _recursive_: False since we have more objects within that we are instantiating
    # without using nested instantiation from hydra
    # workspace = hydra.utils.instantiate(cfg.env.workspace, cfg=cfg, _recursive_=False)
    workspace = Workspace(cfg)
    acc = workspace.evaluate_action_prediction()
    # rewards, infos = workspace.run()
    # print(rewards)
    # print(infos)
    # print(f"Average reward: {np.mean(rewards)}")
    # print(f"Std: {np.std(rewards)}")
    print("ACCURACY: ", acc)


if __name__ == "__main__":
    main()
