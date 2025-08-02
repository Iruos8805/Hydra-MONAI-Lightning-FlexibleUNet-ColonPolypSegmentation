import os
import sys
from pathlib import Path
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


def setup_wandb(cfg: DictConfig, is_sweep: bool = False):
    """Setup Weights & Biases logging"""
    if is_sweep or cfg.wandb.enabled:
        if is_sweep:
            # In sweep mode, wandb is already initialized
            wandb_logger = WandbLogger()
        else:
            # Normal mode with wandb
            wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                name=cfg.get("name", "segmentation_run"),
                config=OmegaConf.to_container(cfg, resolve=True),
                tags=cfg.wandb.tags,
            )
            wandb_logger = WandbLogger()
        return wandb_logger
    return None


def setup_callbacks(cfg: DictConfig, is_sweep: bool = False):
    """Setup training callbacks"""
    callbacks = []

    if not is_sweep:
        # Model checkpointing - let Lightning use default path (this is the fix for wrong checkpoint folder path error)
        checkpoint_callback = ModelCheckpoint(
            monitor=cfg.training.checkpoint.monitor,
            mode=cfg.training.checkpoint.mode,
            save_top_k=cfg.training.checkpoint.save_top_k,
            save_last=cfg.training.checkpoint.save_last,
            filename=cfg.training.checkpoint.filename,
            verbose=True,
        )
        callbacks.append(checkpoint_callback)

        early_stop_callback = EarlyStopping(
            monitor=cfg.training.early_stopping.monitor,
            patience=cfg.training.early_stopping.patience,
            mode=cfg.training.early_stopping.mode,
            min_delta=cfg.training.early_stopping.min_delta,
            verbose=cfg.training.early_stopping.verbose,
        )
        callbacks.append(early_stop_callback)

    return callbacks


def update_config_for_sweep(cfg: DictConfig):
    """Update config with wandb sweep parameters"""
    if wandb.config:
        # Update optimizer settings
        if hasattr(wandb.config, "LR"):
            cfg.training.optimizer.lr = wandb.config.LR
        if hasattr(wandb.config, "OPTIMIZER"):
            optimizer_name = wandb.config.OPTIMIZER.lower()
            if optimizer_name == "adam":
                cfg.training.optimizer._target_ = "torch.optim.Adam"
            elif optimizer_name == "sgd":
                cfg.training.optimizer._target_ = "torch.optim.SGD"
            elif optimizer_name == "adamw":
                cfg.training.optimizer._target_ = "torch.optim.AdamW"

        # Update data settings
        if hasattr(wandb.config, "BATCH_SIZE"):
            cfg.data.batch_size = wandb.config.BATCH_SIZE
        if hasattr(wandb.config, "NUM_WORKERS"):
            cfg.data.num_workers = wandb.config.NUM_WORKERS

        # Update training settings
        if hasattr(wandb.config, "PRECISION"):
            cfg.training.precision = wandb.config.PRECISION


@hydra.main(version_base=None, config_path="config", config_name="config")
def hydra_main(cfg: DictConfig) -> None:
    """Hydra entry point"""
    main(cfg)


def main(cfg: DictConfig = None) -> None:
    """Main training function"""

    # If no config provided, load it manually (for sweep mode)
    if cfg is None:
        from hydra import compose, initialize_config_dir
        from pathlib import Path

        config_dir = Path(__file__).parent / "config"
        initialize_config_dir(config_dir=str(config_dir.absolute()), version_base=None)
        cfg = compose(config_name="config", overrides=["+experiment=sweep"])

    # Print config
    print("=" * 50)
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 50)

    # Set seed for reproducibility (optional)
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed)

    # Check if running in sweep mode
    is_sweep = os.getenv("WANDB_SWEEP_ID") is not None

    if is_sweep:
        print("Running in W&B sweep mode")
        wandb.init()
        update_config_for_sweep(cfg)

    # Setup logging
    logger = setup_wandb(cfg, is_sweep)
    if logger:
        print("W&B logging enabled")
    else:
        print("W&B logging disabled")
        logger = True  # Use default PyTorch Lightning logger

    # Lightning will create its own directories automatically
    print("Using PyTorch Lightning's automatic directory structure")

    # Instantiate data module
    if not cfg.data.image_dir or not cfg.data.mask_dir:
        raise ValueError(
            "Please set image_dir and mask_dir in your config or via command line"
        )

    dm = SegmentationDataModule(
        image_dir=cfg.data.image_dir,
        mask_dir=cfg.data.mask_dir,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        test_size=cfg.data.test_size,
        val_size=cfg.data.val_size,
        random_state=cfg.data.random_state,
        pin_memory=cfg.data.get("pin_memory", True),
    )

    # Instantiate model with optimizer
    optimizer_config = OmegaConf.to_container(cfg.training.optimizer, resolve=True)
    model = SegmentationModel(
        model_cfg=cfg.model,
        optimizer_cfg=optimizer_config,
        scheduler_cfg=cfg.training.get("scheduler", None),
    )

    # Setup callbacks
    callbacks = setup_callbacks(cfg, is_sweep)

    # Create trainer
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        min_epochs=cfg.training.min_epochs,
        max_epochs=cfg.training.max_epochs,
        precision=cfg.training.precision,
        gradient_clip_val=cfg.training.get("gradient_clip_val", None),
        enable_progress_bar=True,
        deterministic=cfg.get("seed") is not None,
    )

    try:
        # Train the model
        trainer.fit(model, dm)

        # Validate
        trainer.validate(model, dm)

        if not is_sweep:

            # Test with best model if available, otherwise use current model
            best_model = None
            if (
                callbacks
                and hasattr(callbacks[0], "best_model_path")
                and callbacks[0].best_model_path
            ):
                try:
                    best_model_path = callbacks[0].best_model_path
                    print(f"Loading best model from: {best_model_path}")

                    best_model = SegmentationModel.load_from_checkpoint(
                        best_model_path,
                        model_cfg=cfg.model,
                        optimizer_cfg=optimizer_config,
                        scheduler_cfg=cfg.training.get(
                            "scheduler", None
                        ),  # try taking hyperparams
                    )
                except Exception as e:
                    print(f"Failed to load best model: {e}")
                    print("Using current model for testing")
                    best_model = model
            else:
                print("No best model checkpoint found, using current model for testing")
                best_model = model

            # Test the model
            print("Running test evaluation...")
            trainer.test(best_model, datamodule=dm)

            # Visualize predictions
            if (
                hasattr(best_model, "test_predictions")
                and best_model.test_predictions.get("inputs") is not None
            ):
                print("Generating visualizations...")
                visualize_batch(
                    best_model.test_predictions["inputs"],
                    best_model.test_predictions["preds"],
                    best_model.test_predictions["targets"],
                )
            else:
                print("No test predictions available for visualization")

    except Exception as e:
        print(f"Training failed: {e}")
        if wandb.run is not None:
            wandb.finish()
        raise

    finally:
        if wandb.run is not None:
            wandb.finish()

    if is_sweep:
        sys.exit(0)


if __name__ == "__main__":
    # Check if running in sweep mode
    is_sweep = os.getenv("WANDB_SWEEP_ID") is not None

    if is_sweep:
        # In sweep mode, call main directly (it will load config internally)
        main()
    else:
        # Normal mode: using Hydra decorator
        hydra_main()
