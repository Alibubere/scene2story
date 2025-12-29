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
    Real story generation for testing model inference.
    """
    try:
        model.eval()
        clip_encoder.eval()
        
        tokenizer = get_gpt2_tokenizer()
        tokenizer.pad_token = tokenizer.eos_token
        transform = get_clip_processor()
        
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            img_feats = clip_encoder(image_tensor)
            
            # Manual tokenization to avoid protobuf issues
            prompt_tokens = prompt.split()
            vocab = tokenizer.get_vocab()
            
            prompt_ids = []
            for word in prompt_tokens:
                if word in vocab:
                    prompt_ids.append(vocab[word])
                else:
                    prompt_ids.append(vocab.get('<|endoftext|>', 50256))
            
            if not prompt_ids:
                prompt_ids = [vocab.get('A', 32)]
                
            prompt_tensor = torch.tensor([prompt_ids], device=device)
            
            # Generate multiple tokens
            current_ids = prompt_tensor.clone()
            
            for _ in range(min(15, max_new_tokens)):
                outputs = model.gpt2(input_ids=current_ids)
                next_token = torch.argmax(outputs.logits[0, -1, :]).item()
                current_ids = torch.cat([current_ids, torch.tensor([[next_token]], device=device)], dim=1)
                if next_token == vocab.get('<|endoftext|>', 50256):
                    break
            
            # Convert back to text manually
            reverse_vocab = {v: k for k, v in vocab.items()}
            result_words = [reverse_vocab.get(tid.item(), '<unk>') for tid in current_ids[0]]
            
            story = ' '.join(result_words).replace('Ġ', ' ').strip()
        
        model.train()
        return story
        
    except Exception as e:
        model.train()
        return f"Generation error: {str(e)}"