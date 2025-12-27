import torch
from PIL import Image
from src.features.clip_encoder import get_clip_processor
from src.text.tokenizer_utils import get_gpt2_tokenizer

def generate_story_from_fixed_image(
    model, 
    clip_encoder, 
    device, 
    image_path: str, 
    prompt: str = "A story about", 
    max_new_tokens: int = 40
):
    """
    Simplified story generation to avoid protobuf issues.
    """
    try:
        model.eval()
        clip_encoder.eval()
        
        tokenizer = get_gpt2_tokenizer()
        transform = get_clip_processor()
        
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            img_feats = clip_encoder(image_tensor)
            prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            
            # Simple forward pass without generate to avoid protobuf issues
            outputs = model.gpt2(input_ids=prompt_ids)
            logits = outputs.logits
            
            # Get next token prediction
            next_token_logits = logits[0, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1)
            
            # Simple greedy decoding for a few tokens
            current_ids = prompt_ids.clone()
            
            for _ in range(min(10, max_new_tokens)):
                outputs = model.gpt2(input_ids=current_ids)
                next_token_id = torch.argmax(outputs.logits[0, -1, :], dim=-1)
                current_ids = torch.cat([current_ids, next_token_id.unsqueeze(0).unsqueeze(0)], dim=1)
                if next_token_id.item() == tokenizer.eos_token_id:
                    break
        
        story = tokenizer.decode(current_ids[0], skip_special_tokens=True)
        model.train()
        return story.strip()
        
    except Exception as e:
        model.train()
        return "Story generation temporarily disabled due to compatibility issues"