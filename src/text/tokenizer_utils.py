from transformers import GPT2Tokenizer
import torch


def get_gpt2_tokenizer(model_name: str = "gpt2"):
    """
    Loads and configure the GPT2 tokenizer for the sequence generations tasks.

    Args:
        model_name (str): The name of the GPT-2 model to load the tokenizer for.

    Returns:
        GPT2Tokenizer: The configured tokenizer instance.
    """

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    new_tokens = ["[IMG]","[SCENE]","[STORY]"]
    tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    return tokenizer

def tokenize_multimodal_entry(tokenizer, scene_text: str, story_text: str, max_length: int = 128):
    """
    Formats text as: [SCENE] {factual description} [STORY] {narrative} <|endoftext|>
    
    Args:
        tokenizer: The configured GPT2 tokenizer
        scene_text: The factual description (e.g., "Two people at a gate in daylight")
        story_text: The creative story
        max_length: Maximum sequence length
    """
    
    # Combine into the "Thinking Structure"
    full_text = f"[SCENE] {scene_text.strip()} [STORY] {story_text.strip()}{tokenizer.eos_token}"

    encoded_output = tokenizer(
        full_text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = encoded_output["input_ids"].squeeze(0)
    attention_mask = encoded_output["attention_mask"].squeeze(0)

    # Labels for Autoregressive training:
    # We want the model to predict everything after the image tokens.
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100  # Ignore padding in loss calculation

    return {
        "input_ids": input_ids, 
        "attention_mask": attention_mask, 
        "labels": labels
    }