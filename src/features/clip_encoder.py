import torch
import clip
from PIL import Image
import torchvision.transforms as transforms


def get_clip_processor():
    """Return CLIP image preprocessing transforms."""
    # CLIP uses 224x224 images with specific normalization
    return transforms.Compose([
        transforms.Resize(224, interpolation=Image.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])


def get_pretrained_clip_encoder(device: torch.device = None):
    """
    Loads a pretrained CLIP Vision Transformer, freezes it, 
    and returns it as a feature extractor.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load CLIP model
    model, _ = clip.load("ViT-B/32", device=device)
    
    # Force to FP32 to avoid dtype mismatch
    model = model.float()
    
    # Extract just the visual encoder
    visual_encoder = model.visual
    
    visual_encoder.eval()
    
    # Freeze all parameters
    for param in visual_encoder.parameters():
        param.requires_grad = False

    return visual_encoder