import os
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from monai.data import Dataset
from custom_dataset import CustomMONAIDataset
from transform import train_transform, val_transform


class SegmentationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        image_dir,
        mask_dir,
        batch_size=4,
        num_workers=4,
        test_size=0.2,
        val_size=0.2,
        random_state=42,
        pin_memory=True,
    ):
        super().__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.pin_memory = pin_memory

    def setup(self, stage=None):
        image_filename = sorted(
            [
                f
                for f in os.listdir(self.image_dir)
                if f.endswith(("jpg", "png", "jpeg"))
            ]
        )
        image_paths = [os.path.join(self.image_dir, f) for f in image_filename]
        mask_paths = [os.path.join(self.mask_dir, f) for f in image_filename]

        assert all(os.path.exists(m) for m in mask_paths), "Some masks are missing"

        data_dicts = [
            {"image": img, "label": lbl} for img, lbl in zip(image_paths, mask_paths)
        ]  # MONAI expects dict

        trainval, test = train_test_split(
            data_dicts, test_size=self.test_size, random_state=self.random_state
        )
        val_ratio = self.val_size / (1.0 - self.test_size)
        train, val = train_test_split(
            trainval, test_size=val_ratio, random_state=self.random_state
        )

        self.train_ds = CustomMONAIDataset(data=train, transform=train_transform)
        self.val_ds = CustomMONAIDataset(data=val, transform=val_transform)
        self.test_ds = CustomMONAIDataset(data=test, transform=val_transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
