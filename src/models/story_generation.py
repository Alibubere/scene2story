import torch
from PIL import Image
from src.features.extract_image_features import get_resnet50_transform
from src.text.tokenizer_utils import get_gpt2_tokenizer

def generate_story_from_fixed_image(
    model, 
    resnet, 
    device, 
    image_path: str, 
    prompt: str = "A story about", 
    max_new_tokens: int = 40
):
    """
    Standard utility to generate a story. 
    Can be used inside train_loop.py or in a standalone test script.
    """
    model.eval()
    resnet.eval()
    
    tokenizer = get_gpt2_tokenizer()
    transform = get_resnet50_transform()
    
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        img_feats = resnet(image_tensor)
        encoded_prompt = tokenizer(prompt, return_tensors="pt").to(device)
    
        output_ids = model.generate(
            img_features=img_feats,
            input_ids=encoded_prompt.input_ids,
            attention_mask=encoded_prompt.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.5, # Stops "marking marking marking"
            no_repeat_ngram_size=2  # Stops "in the in the"
        )
    
    story = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    model.train() 
    return story.strip()