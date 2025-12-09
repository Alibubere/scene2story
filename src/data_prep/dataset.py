from torch.utils.data import Dataset
from src.features.extract_image_features import get_resnet50_transform
import logging
import json
import os
from PIL import Image
from src.text.tokenizer_utils import get_gpt2_tokenizer, tokenize_story
import torch


class StoryImageDataset(Dataset):

    def __init__(self, data_path: str):

        self.data_path = data_path
        self.transform = get_resnet50_transform()
        self.data = []
        self.tokenizer = get_gpt2_tokenizer()
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(['[IMG]'])[0]

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

        token_data = tokenize_story(self.tokenizer, story=story)
        text_ids = token_data["input_ids"].view(-1)
        text_mask = token_data["attention_mask"].view(-1)

        img_token_tensor = torch.tensor([self.image_token_id], dtype=torch.long)
        mask_one_tensor = torch.tensor([1], dtype=torch.long)

        final_input_ids = torch.cat([img_token_tensor, text_ids], dim=0)

        final_attention_mask = torch.cat([mask_one_tensor, text_mask], dim=0)

        final_labels = final_input_ids.clone()
        final_labels[0] = -100

        final_labels[final_attention_mask == 0] = -100

        return (image_tensor, final_input_ids, final_attention_mask, final_labels)
