import torch
from PIL import Image
from src.features.extract_image_features import get_resnet50_transform
from app.dependencies import model, resnet, tokenizer, DEVICE


tranform = get_resnet50_transform()


@torch.no_grad()
def generate_story(
    image: Image.Image,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
):
    image = image.convert("RGB")
    image_tensor = tranform(image).unsqueeze(0).to(DEVICE)

    img_feats = resnet(image_tensor)

    encoded = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    output_ids = model.generate(
        img_features=img_feats,
        input_ids=encoded.input_ids,
        attention_mask=encoded.attention_mask,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=1.4,
        no_repeat_ngram_size=2,
    )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)
