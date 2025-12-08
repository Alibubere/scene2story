from typing import List, Dict, Any
import random
import logging

moods = [
    "calm",
    "cheerful",
    "tense",
    "quiet",
    "busy",
    "lonely",
    "happy",
    "sad",
    "grateful",
    "playful",
]
environment_words = [
    "street",
    "park",
    "market",
    "room",
    "city",
    "field",
    "ocean",
    "stadium",
]
time_of_day = [
    "in the early morning",
    "under the afternoon sun",
    "as the evening settles",
    "late at night",
    "just before the rain",
    "at the hour of midnight",
]
atmosphere_endings = [
    "as the world moves quietly around",
    "while everything else fades into the background",
    "and the truth hung heavy in the silence",
    "until the feeling was all that remained",
    "a memory instantly forged in the mind",
    "with a sense of quiet, profound finality",
    "in a stillness that felt both fragile and absolute",
]


def caption_to_story(caption: str) -> str:
    """
    Convert a single caption into a short 2-3 sentence story.

    Steps:
    1. Clean the caption text.
    2. Add descriptive modifiers (adjectives, mood words, etc.).
    3. Build sentence 1 using the caption as the base.
    4. Add a randomly chosen environmental detail.
    5. Build sentence 2 to create atmosphere.
    6. Return the combined story string.
    """
    if caption:
        if not caption[0].isupper():
            caption = caption[0].upper() + caption[1:]

    caption = caption.strip()

    sentence1 = caption.strip(".") + " " + random.choice(atmosphere_endings).strip(".")

    sentence2 = f"In the background, the {random.choice(environment_words)} feels {random.choice(moods)} {random.choice(time_of_day)}"

    story = sentence1 + "." + " " + sentence2

    return story


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
        random_caption = random.choice(sample["captions"])
        story = caption_to_story(random_caption)

        new_dict: Dict[str, Any] = {
            "image_path": image_path,
            "caption": random_caption,
            "story": story,
        }
        new_samples.append(new_dict)

    logging.info(f"Built {len(new_samples)} samples for story ")

    return new_samples
