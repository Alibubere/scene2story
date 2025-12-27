from torchvision import transforms as T
import torchvision.models as models
import torch


def get_resnet50_transform():
    """Returns a torchvision transform that prepares images for ResNet50."""

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    transform = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    return transform


def get_pretrained_resnet50_encoder(device: torch.device = None):
    """
    Loads a pretrained ResNet50 model, removes the final classification layer (fc),
    and returns it in evaluation mode on the specified device.

    Args:
        device (torch.device, optional): The device (e.g., 'cuda', 'cpu') to move
            the model to. Defaults to checking for CUDA then falling back to CPU.

    Returns:
        torch.nn.Module: The ResNet50 model configured as a feature encoder.
        This module transforms a batch of images into a batch of 2048-dim vectors.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    model.fc = torch.nn.Identity()

    model.eval()

    model.to(device)

    for param in model.parameters():
        param.requires_grad = False

    return model
