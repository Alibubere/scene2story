from torch.utils.data import Dataset
import logging
import json
import os
from PIL import Image
from src.text.tokenizer_utils import get_gpt2_tokenizer, tokenize_multimodal_entry


class StoryImageDataset(Dataset):

    def __init__(self, data_path: str, image_transform,max_length: int = 128):
        """
        Args:
            data_path (str): Path to the .jsonl file containing image paths and stories.
            max_length (int): Max sequence length for the text tokens.
        """
        self.data_path = data_path
        self.transform = image_transform
        self.data = []
        self.tokenizer = get_gpt2_tokenizer()
        self.max_length = max_length

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

        except Exception:
            logging.exception(f"Failed to load or transform image at {image_path}")
            raise RuntimeError(f"Could not load image at path: {image_path}")

        token_data = tokenize_multimodal_entry(
            self.tokenizer, story=story, max_length=self.max_length
        )
        input_ids = token_data["input_ids"]
        attention_mask = token_data["attention_mask"]
        labels = token_data["labels"]

        return (image_tensor, input_ids, attention_mask, labels)
