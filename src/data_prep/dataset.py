from torch.utils.data import Dataset
import logging
import json
import os
from PIL import Image
from src.text.tokenizer_utils import get_gpt2_tokenizer, tokenize_multimodal_entry

# Initialize tokenizer once at module level
_tokenizer = None

def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = get_gpt2_tokenizer()
        # MANDATORY FIX: This stops the "pad_token not set" warnings
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
    return _tokenizer


class StoryImageDataset(Dataset):
    def __init__(self, data_path: str, image_transform, max_length: int = 128, diagnostic_mode: bool = True):
        """
        Args:
            data_path (str): Path to the .jsonl file.
            image_transform: The CLIP processor/transform.
            max_length (int): Max sequence length.
            diagnostic_mode (bool): If True, hides the caption from the input to force CLIP grounding.
        """
        self.data_path = data_path
        self.transform = image_transform
        self.data = []
        self.tokenizer = get_tokenizer()
        self.max_length = max_length
        self.diagnostic_mode = diagnostic_mode 

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file does not exist {self.data_path}")

        with open(self.data_path, "r") as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        sample = self.data[index]
        image_path = sample["image_path"]
        story_text = sample.get("story", sample.get("caption", ""))

        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.transform(image)
        except Exception:
            logging.exception(f"Failed to load image at {image_path}")
            raise RuntimeError(f"Could not load image at path: {image_path}")

        scene_text = "" if self.diagnostic_mode else sample.get("caption", "")

        token_data = tokenize_multimodal_entry(
            self.tokenizer, 
            scene_text=scene_text, 
            story_text=story_text, 
            max_length=self.max_length
        )

        return (
            image_tensor, 
            token_data["input_ids"], 
            token_data["attention_mask"], 
            token_data["labels"]
        )