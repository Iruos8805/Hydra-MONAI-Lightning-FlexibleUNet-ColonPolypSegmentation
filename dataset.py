import os
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from monai.data import Dataset
from custom_dataset import CustomMONAIDataset


class SegmentationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        image_dir,
        mask_dir,
        normalize,
        image_size,
        test_size,
        val_size,
        random_state,
        train,
        val,
        test,
        train_transform=None,
        val_transform=None,
    ):
        super().__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.normalize = normalize
        self.image_size = image_size
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

        self.train_cfg = train or {}
        self.val_cfg = val or {}
        self.test_cfg = test or {}

        self.train_transform = train_transform
        self.val_transform = val_transform

    def setup(self, stage=None):
        image_filenames = sorted(
            [
                f
                for f in os.listdir(self.image_dir)
                if f.endswith(("jpg", "jpeg", "png"))
            ]
        )
        image_paths = [os.path.join(self.image_dir, f) for f in image_filenames]
        mask_paths = [os.path.join(self.mask_dir, f) for f in image_filenames]

        assert all(os.path.exists(p) for p in mask_paths), "Some masks are missing."

        data_dicts = [{"image": i, "label": m} for i, m in zip(image_paths, mask_paths)]

        trainval, test = train_test_split(
            data_dicts, test_size=self.test_size, random_state=self.random_state
        )
        val_ratio = self.val_size / (1.0 - self.test_size)
        train, val = train_test_split(
            trainval, test_size=val_ratio, random_state=self.random_state
        )

        self.train_ds = CustomMONAIDataset(train, transform=self.train_transform)
        self.val_ds = CustomMONAIDataset(val, transform=self.val_transform)
        self.test_ds = CustomMONAIDataset(test, transform=self.val_transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.train_cfg.get("batch_size", 4),
            shuffle=self.train_cfg.get("shuffle", True),
            num_workers=self.train_cfg.get("num_workers", 4),
            pin_memory=self.train_cfg.get("pin_memory", True),
            drop_last=self.train_cfg.get("drop_last", True),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.val_cfg.get("batch_size", 4),
            shuffle=self.val_cfg.get("shuffle", False),
            num_workers=self.val_cfg.get("num_workers", 4),
            pin_memory=self.val_cfg.get("pin_memory", True),
            drop_last=self.val_cfg.get("drop_last", False),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.test_cfg.get("batch_size", 4),
            shuffle=self.test_cfg.get("shuffle", False),
            num_workers=self.test_cfg.get("num_workers", 4),
            pin_memory=self.test_cfg.get("pin_memory", True),
            drop_last=self.test_cfg.get("drop_last", False),
        )
