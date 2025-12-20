import torch
import yaml
from src.models.multimodel_gpt2 import MultimodelGPT2
from src.features.extract_image_features import get_pretrained_resnet50_encoder
from src.text.tokenizer_utils import get_gpt2_tokenizer


with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# model config
model_config = config["model"]
gpt2_model_name = model_config["gpt2_type"]
num_image_tokens = model_config["num_img_tokens"]
num_unfreeze_layers = model_config["num_unfreeze_layers"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = get_gpt2_tokenizer()

resnet = get_pretrained_resnet50_encoder(DEVICE)
resnet.eval()


model = MultimodelGPT2(
    gpt2_model_name=gpt2_model_name,
    num_img_tokens=num_image_tokens,
    num_unfreeze_layers=num_unfreeze_layers,
).to(DEVICE)

checkpoint = torch.load("checkpoint/best.pth",map_location=DEVICE)
model.load_state_dict(checkpoint["model_state"])
model.eval()