<div align="center">

# 🎨 Scene2Story 📸→📖

<h3>✨ Transform Images into Compelling Narratives using Deep Learning ✨</h3>

<p align="center">
  <img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=22&duration=3000&pause=1000&color=6366F1&center=true&vCenter=true&width=600&lines=AI-Powered+Story+Generation;ResNet50+%2B+GPT-2+Architecture;From+Pixels+to+Prose" alt="Typing SVG" />
</p>

---

### 🚀 Tech Stack

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/🤗_Transformers-Latest-FFD21E?style=for-the-badge)
![CLIP](https://img.shields.io/badge/CLIP-Vision_Encoder-FF6B6B?style=for-the-badge)
![GPT-2](https://img.shields.io/badge/GPT--2-Language_Model-4ECDC4?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_UI-FF4B4B?style=for-the-badge)

### 📊 Project Status

![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)
![Maintenance](https://img.shields.io/badge/Maintained-Yes-brightgreen?style=for-the-badge)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey?style=for-the-badge)
![Code Style](https://img.shields.io/badge/Code_Style-Black-000000?style=for-the-badge)

---

### 🧭 Quick Navigation

<p align="center">
  <a href="#-features">🌟 Features</a> •
  <a href="#-installation">📦 Installation</a> •
  <a href="#-usage">🚀 Usage</a> •
  <a href="#️-configuration">⚙️ Configuration</a> •
  <a href="#-model-training">🎯 Training</a>
</p>

</div>

<br>

## 🎯 Overview

<div align="center">
<table>
<tr>
<td width="50%">

**🔍 What it does:**
- Analyzes images using CLIP vision encoder
- Extracts meaningful visual features
- Generates creative narratives
- Combines CLIP + GPT-2 architectures
- Interactive web interface

</td>
<td width="50%">

**🎪 Perfect for:**
- Creative writing assistance
- Educational storytelling
- Content generation
- AI research projects

</td>
</tr>
</table>
</div>

> **Scene2Story** is an AI-powered system that transforms images into compelling narratives by seamlessly combining computer vision and natural language processing technologies.

## ✨ Features

- 🖼️ **Image Feature Extraction** - CLIP-based visual encoding
- 📝 **Story Generation** - GPT-2 powered narrative creation
- 🔄 **Custom Dataset Pipeline** - Flickr30k integration
- 🧠 **Model Training** - Complete training pipeline with checkpointing
- ⚙️ **YAML Configuration** - Easy parameter tuning
- 📊 **Logging System** - Comprehensive tracking
- 🎨 **Multimodal Architecture** - Custom image token integration
- 💾 **Model Persistence** - Automatic checkpoint saving and loading
- 🌐 **Web Interface** - Streamlit-based interactive UI
- 🚀 **FastAPI Support** - Ready for API deployment

## 📦 Installation

### Prerequisites

- Python 3.8+
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# Clone repository
git clone https://github.com/Alibubere/scene2story.git
cd scene2story

# Install dependencies
pip install -r requirements.txt

# Run the web interface
streamlit run app/ui.py
```

### Dataset Download

📥 **[Download Flickr30k Dataset](https://huggingface.co/datasets/nlphuji/flickr30k/resolve/main/flickr30k-images.zip?download=true)** (Auto-download)

Extract the downloaded zip file to `data/raw/flickr30k-images/`

## 🚀 Usage

### Training the Model

```bash
# Start training from scratch
python main.py

# Training will automatically resume from latest checkpoint if available
```

### Web Interface Usage

```bash
# Launch the interactive web interface
streamlit run app/ui.py
```

### Quick Start Example

```python
from src.data_prep.dataset import StoryImageDataset
from src.models.multimodel_gpt2 import MultimodelGPT2
from src.features.clip_encoder import get_pretrained_clip_encoder

# Load dataset
dataset = StoryImageDataset("data/processed/stories_train.jsonl")

# Initialize model
model = MultimodelGPT2(
    gpt2_model_name="gpt2",
    num_img_tokens=4,
    num_unfreeze_layers=4
)

# Load trained weights
checkpoint = torch.load("checkpoints/best.pth")
model.load_state_dict(checkpoint["model_state"])
```

## 📁 Project Structure

```
scene2story/
├── app/
│   └── ui.py                    # Streamlit web interface
├── checkpoints/
│   ├── best.pth                 # Best model checkpoint
│   └── latest.pth               # Latest training checkpoint
├── configs/
│   └── config.yaml              # Configuration file
├── data/
│   ├── processed/               # Processed datasets
│   │   ├── stories_train.jsonl  # Training data
│   │   └── stories_val.jsonl    # Validation data
│   └── raw/                     # Raw Flickr30k data
│       ├── flickr30k-images/    # Image files
│       └── flickr_annotations_30k.csv
├── src/
│   ├── data_prep/
│   │   ├── dataloader.py        # Data loading utilities
│   │   ├── dataset.py           # PyTorch dataset
│   │   ├── flickr_loader.py     # Flickr data processing
│   │   ├── story_generator.py   # Story creation logic
│   │   └── save_story_dataset.py
│   ├── features/
│   │   ├── clip_encoder.py      # CLIP vision encoder
│   │   └── extract_image_features.py  # Feature extraction utilities
│   ├── models/
│   │   ├── multimodel_gpt2.py   # Main multimodal model
│   │   ├── story_generation.py  # Generation utilities
│   │   ├── train.py             # Training script
│   │   ├── train_loop.py        # Training loop logic
│   │   └── training_utils.py    # Training utilities
│   └── text/
│       └── tokenizer_utils.py   # GPT-2 tokenization
├── logs/
│   └── Pipeline.log             # Training logs
├── requirements.txt             # Python dependencies
├── LICENCE                      # MIT License
└── main.py                      # Training entry point
```

## ⚙️ Configuration

Edit `configs/config.yaml`:

```yaml
paths:
  annotations_csv: "data/raw/flickr_annotations_30k.csv"
  images_dir: "data/raw/flickr30k-images"

data:
  use_split: "train"   # 'train', 'val', 'test', or 'all'
  num_preview: 3

clean_paths:
  save_dir: "data/processed"

model:
  gpt2_type: "gpt2"           # 'gpt2' or 'gpt2-medium'
  d_model: 768                # GPT-2 hidden size
  num_img_tokens: 4           # Number of image tokens
  num_unfreeze_layers: 4      # Trainable transformer layers
  dropout: 0.1

training:
  num_epochs: 20
  batch_size: 8
  lr: 0.00005
  weight_decay: 0.005
  resume_from_checkpoint: True
  use_amp: False
  max_grad_norm: 1.0
  early_stop_patience: 7
```

## 🛠️ Tech Stack

- **Deep Learning**: PyTorch, torchvision
- **NLP**: Hugging Face Transformers (GPT-2)
- **Vision**: CLIP (OpenAI)
- **Data**: Pandas, PIL
- **Config**: PyYAML
- **Web Interface**: Streamlit
- **API**: FastAPI, Uvicorn
- **Training**: Mixed precision training, automatic checkpointing
- **Logging**: Comprehensive training metrics

## 🌐 Web Interface

The project includes a user-friendly Streamlit web interface:

- **Image Upload**: Drag and drop or browse for images
- **Custom Prompts**: Add optional story prompts
- **Real-time Generation**: Generate stories with adjustable parameters
- **Interactive Display**: View generated stories alongside input images

### Launch Web Interface

```bash
streamlit run app/ui.py
```

## 🎯 Model Training

The model supports:
- **Automatic Checkpointing**: Saves best and latest model states
- **Resume Training**: Automatically resumes from the latest checkpoint
- **Mixed Precision**: Efficient GPU memory usage
- **Validation Tracking**: Monitors validation loss for best model selection
- **Comprehensive Logging**: Detailed training metrics in `logs/Pipeline.log`

### Training Progress

Monitor training progress through:
- Real-time loss tracking every 250 batches
- Validation loss evaluation after each epoch
- Generated story samples for quality assessment
- Automatic best model saving based on validation performance

## 👨💻 Author

**Alibubere**

[![GitHub](https://img.shields.io/badge/GitHub-Alibubere-181717?style=flat&logo=github)](https://github.com/Alibubere)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Alibubere-0A66C2?style=flat&logo=linkedin)](https://www.linkedin.com/in/mohammad-ali-bubere-a6b830384/)
[![Email](https://img.shields.io/badge/Email-Contact-D14836?style=flat&logo=gmail&logoColor=white)](mailto:alibubere989@gmail.com)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Feel free to:

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## ⭐ Show Your Support

Give a ⭐️ if you like this project!

## 📮 Contact

For questions or feedback, please open an issue on GitHub.

---

<div align="center">
Made with ❤️ by Alibubere
</div>
