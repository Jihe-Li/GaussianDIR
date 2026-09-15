import os

import hydra
from omegaconf import DictConfig, OmegaConf

import registrators


@hydra.main(version_base=None, config_path="configs", config_name="gaussian.yaml")
def main(cfg: DictConfig) -> None:
    cfg.save_dir = os.path.join("outputs", cfg.exp_name, 'Case%d' % cfg.dataset.case_idx)
    os.makedirs(cfg.save_dir, exist_ok=True)
    with open(os.path.join(cfg.save_dir, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    cfg.training = True
    registrator = registrators.__dict__[cfg.registrator.type](cfg)
    registrator.train()
    registrator.close()


if __name__ == "__main__":
    main()
