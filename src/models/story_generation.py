import torch
from PIL import Image
from src.features.extract_image_features import get_resnet50_transform
from src.text.tokenizer_utils import get_gpt2_tokenizer


def generate_story_from_fixed_image(
    model, resnet, device, image_path: str, max_length: int = 25
):
    """Generate story from a fixed image for monitoring training progress."""
    
    model.eval()
    tokenizer = get_gpt2_tokenizer()
    transform = get_resnet50_transform()
    
    # Load and process image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get image features
    with torch.no_grad():
        img_feats = resnet(image_tensor)
    
    # Start with image token
    img_token_id = tokenizer.convert_tokens_to_ids(['[IMG]'])[0]
    input_ids = torch.tensor([[img_token_id]], device=device)
    
    generated_tokens = []
    
    with torch.no_grad():
        for _ in range(max_length):
            # Create attention mask
            attention_mask = torch.ones_like(input_ids)
            
            # Get model predictions
            logits = model(img_feats, input_ids, attention_mask)
            
            # Get next token with balanced sampling
            logits_last = logits[0, -1, :] / 0.9  # moderate temperature
            
            # Top-k sampling (keep only top 100 tokens for more diversity)
            top_k = 100
            top_k_logits, top_k_indices = torch.topk(logits_last, top_k)
            probs = torch.softmax(top_k_logits, dim=-1)
            selected_idx = torch.multinomial(probs, 1)
            next_token_id = top_k_indices[selected_idx].unsqueeze(0)
            
            # Stop if EOS token or repetition
            token_id = next_token_id.item()
            if token_id == tokenizer.eos_token_id:
                break
            
            # Stop if same token repeated 3+ times consecutively
            if len(generated_tokens) >= 2 and all(t == token_id for t in generated_tokens[-2:]):
                break
                
            generated_tokens.append(token_id)
            input_ids = torch.cat([input_ids, next_token_id], dim=1)
    
    # Decode story (skip image token)
    story = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    model.train()
    return story.strip()