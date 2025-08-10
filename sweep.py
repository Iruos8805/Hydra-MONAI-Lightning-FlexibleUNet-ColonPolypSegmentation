import wandb
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from omegaconf import OmegaConf, DictConfig
from model import SegmentationModel
from dataset import SegmentationDataModule
import hydra
from hydra.utils import instantiate
import yaml

torch.set_float32_matmul_precision("medium")


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed)

    logger = WandbLogger(**cfg.wandb) if cfg.wandb_logging.enabled else None

    dm = SegmentationDataModule(
        **cfg.data,
        train_transform=instantiate(cfg.transforms.train),
        val_transform=instantiate(cfg.transforms.val),
    )

    optimizer_cfg = OmegaConf.to_container(cfg.training.optimizer, resolve=True)
    model = SegmentationModel(
        model_cfg=cfg.model,
        optimizer_cfg=optimizer_cfg,
        scheduler_cfg=cfg.training.get("scheduler", None),
    )

    trainer = pl.Trainer(
        logger=logger,
        **cfg.training.trainer,
        enable_progress_bar=True,
        deterministic=cfg.get("seed") is not None,
    )

    trainer.fit(model, dm)
    trainer.validate(model, dm)

    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()