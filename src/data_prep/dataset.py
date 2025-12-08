from torch.utils.data import Dataset
from src.features.extract_image_features import get_resnet50_transform
import logging
import json
import os
from PIL import Image


class StoryImageDataset(Dataset):

    def __init__(self, data_path: str):

        self.data_path = data_path
        self.transform = get_resnet50_transform()
        self.data = []

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file does not exist {self.data_path}")

        if not self.data_path.endswith(".jsonl"):
            logging.error(f"Data path must ends with .jsonl but got {self.data_path}")
            raise ValueError(
                f"Data path must ends with .jsonl but got {self.data_path}"
            )

        with open(self.data_path, "r") as f:

            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):

        sample = self.data[index]
        image_path = sample["image_path"]
        story = sample["story"]

        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.transform(image)

            return image_tensor, story
        except Exception:
            logging.exception(f"Failed to load or transform image at {image_path}")
            raise RuntimeError(f"Could not load image at path: {image_path}")
