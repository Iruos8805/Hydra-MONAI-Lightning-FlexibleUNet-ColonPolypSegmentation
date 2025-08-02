import numpy as np
from PIL import Image
import torch
from monai.data import Dataset


class CustomMONAIDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]

        if self.transform:
            # sample with paths for MONAI transforms
            sample = {
                "image": item["image"],  # File path
                "label": item["label"],  # File path
            }
            sample = self.transform(sample)
            # Returning dictionary instead of tuple (which was the earlier approach)
            return sample  # This contains {"image": tensor, "label": tensor}
        else:
            # Manual loading when no transforms
            image = np.array(Image.open(item["image"]).convert("RGB"))
            mask = np.array(Image.open(item["label"]).convert("L"))

            # Convert to tensors
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0) / 255.0

            # Return dictionary instead of tuple (which was the earlier approach)
            return {"image": image, "label": mask}
