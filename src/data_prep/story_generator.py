from typing import List, Dict, Any
import random
import logging

NEUTRAL_FOLLOWUPS = [
    "The image shows a simple moment.",
    "The scene captures an everyday situation.",
    "This appears to be an ordinary scene.",
    "The image presents a clear visual moment."
]


def caption_to_story(caption: str) -> str:
    """
    Convert a single caption into a short 2-3 sentence story.

    Steps:
    1. Clean the caption text.
    2. Build sentence 1 using the caption as the base.
    3. Build sentence 2 to create atmosphere.
    4. Return the combined story string.
    """
    if caption is None or len(caption.strip()) == 0:
        return ""

    caption = caption.strip()
    caption = caption.rstrip(".")

    caption = caption[0].upper() + caption[1:]
    
    return caption + "."

def build_story_dataset(samples: List[Dict]) -> List[Dict]:
    """
    Build the full story dataset from parsed samples.

    Steps:
    1. Loop over all samples.
    2. Take one caption from each (e.g., captions[0]).
    3. Convert that caption into a generated story.
    4. Return a list of dictionaries:
        {
            'image_path': ...,
            'caption': ...,
            'story': ...
        }
    """
    new_samples: List[Dict[str, Any]] = []

    for sample in samples:
        image_path = sample["image_path"]
        caption = sample["captions"][0]
        story = caption_to_story(caption)

        new_samples.append({
            "image_path": image_path,
            "caption": caption,
            "story": story,
            "img_id": sample.get("img_id"),
            "split": sample.get("split"),
        })


    logging.info(f"Built {len(new_samples)} samples for story ")

    return new_samples
