# Scene2Story 📸→📖

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/🤗_Transformers-Latest-FFD21E?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)
![Maintenance](https://img.shields.io/badge/Maintained-Yes-brightgreen?style=for-the-badge)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey?style=for-the-badge)
![Code Style](https://img.shields.io/badge/Code_Style-Black-000000?style=for-the-badge)

**Transform images into compelling narratives using deep learning**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Configuration](#️-configuration)

</div>

---

## 🎯 Overview

Scene2Story is an AI-powered system that generates creative stories from images by combining computer vision and natural language processing. It leverages ResNet50 for visual feature extraction and GPT-2 for coherent story generation.

## ✨ Features

- 🖼️ **Image Feature Extraction** - ResNet50-based visual encoding
- 📝 **Story Generation** - GPT-2 powered narrative creation
- 🔄 **Custom Dataset Pipeline** - Flickr30k integration
- ⚙️ **YAML Configuration** - Easy parameter tuning
- 📊 **Logging System** - Comprehensive tracking
- 🎨 **Special Token Handling** - Custom [IMG] token support

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
pip install torch torchvision transformers pillow pandas pyyaml
```

## 🚀 Usage

```bash
python main.py
```

### Quick Start Example

```python
from src.data_prep.dataset import StoryImageDataset

# Load dataset
dataset = StoryImageDataset("data/processed/stories_train.jsonl")

# Get sample
image, input_ids, attention_mask, labels = dataset[0]
```

## 📁 Project Structure

```
scene2story/
├── configs/
│   └── config.yaml              # Configuration file
├── data/
│   ├── processed/               # Processed datasets
│   └── raw/                     # Raw Flickr30k data
├── src/
│   ├── data_prep/
│   │   ├── dataset.py           # PyTorch dataset
│   │   ├── flickr_loader.py     # Data loading utilities
│   │   ├── story_generator.py   # Story creation logic
│   │   └── save_story_dataset.py
│   ├── features/
│   │   └── extract_image_features.py  # ResNet50 feature extraction
│   └── text/
│       └── tokenizer_utils.py   # GPT-2 tokenization
├── logs/                        # Training logs
└── main.py                      # Entry point
```

## ⚙️ Configuration

Edit `configs/config.yaml`:

```yaml
paths:
  annotations_csv: "data/raw/flickr_annotations_30k.csv"
  images_dir: "data/raw/flickr30k-images"

data:
  use_split: "train"
  num_preview: 5

clean_paths:
  save_dir: "data/processed"
```

## 🛠️ Tech Stack

- **Deep Learning**: PyTorch, torchvision
- **NLP**: Hugging Face Transformers (GPT-2)
- **Vision**: ResNet50 (pretrained)
- **Data**: Pandas, PIL
- **Config**: PyYAML

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
