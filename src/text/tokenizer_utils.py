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

    tokenizer.add_special_tokens({"additional_special_tokens": ["[IMG]"]})

    return tokenizer


def tokenize_story(tokenizer, story: str, max_length: int = 128):
    """
    Tokenizes a single story string and creates input_ids, attention_mask, and labels.

    Args:
        tokenizer (GPT2Tokenizer): The configured tokenizer
        story (str): The story text to tokenize.
        max_length (int): The maximum sequence length for truncation and padding

    Returns:
        dict: A dictionary containing the tokenized tensors.
    """

    story_with_eos = story + tokenizer.eos_token

    encoded_output = tokenizer(
        story_with_eos,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = encoded_output["input_ids"].squeeze(0)
    attention_mask = encoded_output["attention_mask"].squeeze(0)

    labels = input_ids.clone()

    labels[attention_mask == 0] = -100

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
