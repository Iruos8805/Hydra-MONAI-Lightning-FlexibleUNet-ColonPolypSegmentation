import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

from model import SegmentationModel
from dataset import SegmentationDataModule
from plot import visualize_batch

torch.set_float32_matmul_precision("medium")


def setup_wandb(cfg: DictConfig):
    if cfg.wandb_logging.enabled:
        return WandbLogger(**cfg.wandb)
    return None


def get_best_model_path(callbacks: list) -> str:
    for cb in callbacks:
        if isinstance(cb, ModelCheckpoint) and cb.best_model_path:
            return cb.best_model_path
    raise ValueError("No valid best_model_path found in ModelCheckpoint callbacks.")


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    if cfg.get("seed"):
        pl.seed_everything(cfg.seed)

    logger = setup_wandb(cfg)
    callbacks_config = instantiate(cfg.callbacks)
    callbacks = list(callbacks_config["callbacks"])

    if not cfg.data.image_dir or not cfg.data.mask_dir:
        raise ValueError("Set image_dir and mask_dir in config")

    dm = SegmentationDataModule(
        **cfg.data,
        train_transform=instantiate(cfg.transforms.train),
        val_transform=instantiate(cfg.transforms.val),
    )

    optimizer_config = OmegaConf.to_container(cfg.training.optimizer, resolve=True)
    model = SegmentationModel(
        model_cfg=cfg.model,
        optimizer_cfg=optimizer_config,
        scheduler_cfg=cfg.training.get("scheduler", None),
    )

    trainer = pl.Trainer(
        logger=logger,
        callbacks= callbacks,
        **cfg.training.trainer,
        enable_progress_bar=True,
        deterministic=cfg.get("seed") is not None,
    )

    trainer.fit(model, dm)
    trainer.validate(model, dm)

    best_model = model
    try:
        best_model_path = get_best_model_path(callbacks)
        print(f"Loading best model from: {best_model_path}")
        best_model = best_model = SegmentationModel.load_from_checkpoint(best_model_path)

    except Exception as e:
        print(f"Could not load best model: {e}")

    trainer.test(best_model, datamodule=dm)

    if (
        hasattr(best_model, "test_predictions")
        and best_model.test_predictions.get("inputs") is not None
    ):
        visualize_batch(
            best_model.test_predictions["inputs"],
            best_model.test_predictions["preds"],
            best_model.test_predictions["targets"],
        )

    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
