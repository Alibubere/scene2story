from transformers import CLIPVisionModel , CLIPImageProcessor
import torch


def get_clip_processor():
    """Return the official CLIP processor for image normalization/resizing."""
    return CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")


def get_pretrained_clip_encoder(device: torch.device = None):
    """
    Loads a pretrained CLIP Vision Transformer, freezes it, 
    and returns it as a feature extractor.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # We use ViT-B/32. It's lightweight and fits on an RTX 4060 easily.
    model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
    
    model.eval()
    model.to(device)

    for param in model.parameters():
        param.requires_grad = False

    return model