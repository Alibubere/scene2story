from transformers import GPT2Tokenizer
import torch


def get_gpt2_tokenizer(model_name:str = "gpt2"):
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

    return tokenizer