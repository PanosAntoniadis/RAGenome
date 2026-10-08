import hydra
import torch
from hydra.utils import instantiate
from lightning import seed_everything
from omegaconf import DictConfig

from ragenome.utils.hydra_utils import instantiate_callbacks


@hydra.main(version_base=None, config_path="../configs", config_name="pretraining")
def main(cfg: DictConfig) -> None:
    seed_everything(12345, workers=True)
    torch.set_float32_matmul_precision("medium")

    datamodule = instantiate(cfg.data)
    model = instantiate(cfg.model)

    callbacks = instantiate_callbacks(cfg.get("callbacks"))
    logger = instantiate(cfg.logger) if cfg.get("logger") else None

    trainer = instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))


if __name__ == "__main__":
    main()
