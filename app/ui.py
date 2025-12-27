import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import torch
from PIL import Image
import yaml
from src.models.multimodel_gpt2 import MultimodelGPT2
from src.features.extract_image_features import get_pretrained_resnet50_encoder, get_resnet50_transform
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
        resnet = get_pretrained_resnet50_encoder(DEVICE)
        resnet.eval()
        transform = get_resnet50_transform()
        
        model = MultimodelGPT2(
            gpt2_model_name=gpt2_model_name,
            num_img_tokens=num_image_tokens,
            num_unfreeze_layers=num_unfreeze_layers,
        ).to(DEVICE)
        
        checkpoint = torch.load(best_model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state"], strict=False)
        model.eval()
        
        return model, resnet, tokenizer, transform, DEVICE
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, None, None

@torch.no_grad()
def generate_story(image, prompt, max_new_tokens, model, resnet, tokenizer, transform, device):
    """Generate story from image and prompt"""
    image = image.convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    img_feats = resnet(image_tensor)
    
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    
    output_ids = model.generate(
        img_features=img_feats,
        input_ids=encoded.input_ids,
        attention_mask=encoded.attention_mask,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.4,
        no_repeat_ngram_size=2,
    )
    
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

st.title("🖼️ Scene → Story Generator")
st.caption("Upload an image. Get a short narrative.")

# Load model
model, resnet, tokenizer, transform, device = load_model()

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
                resnet=resnet,
                tokenizer=tokenizer,
                transform=transform,
                device=device
            )
            
            st.subheader("📖 Generated Story")
            st.write(story)
            
            # Display the uploaded image
            st.subheader("🖼️ Input Image")
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
        except Exception as e:
            st.error(f"Error generating story: {str(e)}")
