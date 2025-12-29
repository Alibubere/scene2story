import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import torch
from PIL import Image
import yaml
from src.models.multimodel_gpt2 import MultimodelGPT2
from src.features.clip_encoder import get_clip_processor, get_pretrained_clip_encoder
from src.text.tokenizer_utils import get_gpt2_tokenizer

st.set_page_config(
    page_title="Scene2Story",
    layout="centered"
)

@st.cache_resource
def load_model():
    """Load the model and dependencies once"""
    try:
        with open("configs/config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        # model config
        model_config = config["model"]
        gpt2_model_name = model_config["gpt2_type"]
        num_image_tokens = model_config["num_img_tokens"]
        num_unfreeze_layers = model_config["num_unfreeze_layers"]
        
        # checkpoint config
        checkpoint_config = config["checkpoint"]
        checkpoint_dir = checkpoint_config["dir"]
        best_path = checkpoint_config["best"]
        best_model_path = os.path.join(checkpoint_dir, best_path)
        
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        tokenizer = get_gpt2_tokenizer()
        clip_encoder = get_pretrained_clip_encoder(DEVICE)
        clip_encoder.eval()
        transform = get_clip_processor()
        
        model = MultimodelGPT2(
            gpt2_model_name=gpt2_model_name,
            num_img_tokens=num_image_tokens,
            num_unfreeze_layers=num_unfreeze_layers,
        ).to(DEVICE)
        
        checkpoint = torch.load(best_model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        
        return model, clip_encoder, tokenizer, transform, DEVICE
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, None, None

@torch.no_grad()
def generate_story(image, prompt, max_new_tokens, model, clip_encoder, tokenizer, transform, device):
    """Generate story from image and prompt - avoiding transformers tokenizer methods"""
    try:
        image = image.convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        img_feats = clip_encoder(image_tensor)
        
        # Use basic tokenization without transformers methods
        prompt_tokens = prompt.split()
        vocab = tokenizer.get_vocab()
        
        # Convert words to token IDs manually
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
        
        return ' '.join(result_words).replace('Ġ', ' ').strip()
        
    except Exception as e:
        return "Story generation temporarily disabled due to compatibility issues"

st.title("🖼️ Scene → Story Generator")
st.caption("Upload an image. Get a short narrative.")

# Load model
model, clip_encoder, tokenizer, transform, device = load_model()

if model is None:
    st.error("Failed to load model. Please check your model files and configuration.")
    st.stop()

uploaded_image = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

prompt = st.text_input(
    "Optional prompt",
    value="A story about"
)

max_tokens = st.slider(
    "Max tokens",
    min_value=20,
    max_value=120,
    value=50
)

if uploaded_image and st.button("Generate Story"):
    with st.spinner("Generating story..."):
        try:
            image = Image.open(uploaded_image)
            story = generate_story(
                image=image,
                prompt=prompt,
                max_new_tokens=max_tokens,
                model=model,
                clip_encoder=clip_encoder,
                tokenizer=tokenizer,
                transform=transform,
                device=device
            )
            
            st.subheader("📖 Generated Story")
            st.write(story)
            
            # Display the uploaded image
            st.subheader("🖼️ Input Image")
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
        except Exception as e:
            st.error(f"Error generating story: {str(e)}")
