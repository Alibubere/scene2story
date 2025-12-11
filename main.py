import yaml
import logging
import os
import torch
from src.data_prep.flickr_loader import (
    load_flickr_annotations,
    build_samples_list,
    parse_captions_column,
)
from src.data_prep.story_generator import build_story_dataset
from src.data_prep.save_story_dataset import save_clean_dataset
from src.data_prep.dataset import StoryImageDataset
from src.text.tokenizer_utils import get_gpt2_tokenizer
from src.models.decoder import ImageConditionedTransformerDecoder
from src.features.extract_image_features import get_pretrained_resnet50_encoder
from src.data_prep.dataloader import get_dataloader

def logging_setup():

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    filename = "Pipeline.log"

    full_path = os.path.join(log_dir, filename)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(full_path), logging.StreamHandler()],
    )
    logging.info("Logging initialize successfully")


def main():
    logging_setup()

    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # paths config
    paths = config["paths"]
    csv_path = paths["annotations_csv"]
    images_dir = paths["images_dir"]

    # data config
    data = config["data"]
    split = data["use_split"]
    num_preview = data["num_preview"]

    # clean path config
    clean_path = config["clean_paths"]
    save_dir = clean_path["save_dir"]

    # model config
    model_config = config["model"]
    d_model = model_config["d_model"]
    n_heads = model_config["n_heads"]
    num_layers = model_config["num_layers"]
    dim_feedforward = model_config["dim_feedforward"]
    max_seq_len = model_config["max_seq_len"]
    dropout = model_config["dropout"]

    train_data_path = "data/processed/stories_train.jsonl"
    tokenizer = get_gpt2_tokenizer()
    vocab_size = len(tokenizer)
    df = load_flickr_annotations(csv_path=csv_path, split=split)
    df = parse_captions_column(df)
    samples = build_samples_list(df, images_dir)
    logging.info(f"Total samples: {len(samples)}")
    stories = build_story_dataset(samples=samples)
    save_clean_dataset(stories, save_dir, split)
    dataset = StoryImageDataset(train_data_path)
    
    # Extract image features
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resnet = get_pretrained_resnet50_encoder(device)

    model = ImageConditionedTransformerDecoder(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        max_seq_len=max_seq_len,
        dropout=dropout,
    ).to(device)

    # sanity check one batch
    loader = get_dataloader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0,   # keep 0 for now
)
    images, input_ids, attn_mask, labels = next(iter(loader))
    images = images.to(device)
    input_ids = input_ids.to(device)
    attn_mask = attn_mask.to(device)
    labels = labels.to(device)
    print(images.shape, input_ids.shape, attn_mask.shape, labels.shape)

    with torch.no_grad():
       img_features = resnet(images)
    
    logits = model(img_features, input_ids, attn_mask)
    print(logits.shape)

if __name__ == "__main__":
    main()
