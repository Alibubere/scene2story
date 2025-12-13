import yaml
import logging
import os
import torch
from torch.cuda.amp import GradScaler
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
from src.models.training_utils import (
    get_optimizer,
    get_lr_scheduler,
)
from src.models.train_loop import train_loop


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

    # Training config
    training_config = config["training"]
    num_epochs = training_config["num_epochs"]
    batch_size = training_config["batch_size"]
    lr = training_config["lr"]
    weight_decay = training_config["weight_decay"]
    num_workers = training_config["num_workers"]
    resume = training_config["resume_from_checkpoint"]
    use_amp = training_config["use_amp"]

    # Checkpoint config
    checkpoint = config["checkpoint"]
    checkpoint_dir = checkpoint["dir"]
    latest = checkpoint["latest"]
    best = checkpoint["best"]
    latest_path = os.path.join(checkpoint_dir, latest)
    best_path = os.path.join(checkpoint_dir, best)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Fixed image for monitoring
    fixed_image_path = config.get("monitoring", {}).get("fixed_image_path")

    train_data_path = "data/processed/stories_train.jsonl"
    val_data_path = "data/processed/stories_val.jsonl"
    tokenizer = get_gpt2_tokenizer()
    vocab_size = len(tokenizer)

    train_df = load_flickr_annotations(csv_path=csv_path, split=split)
    val_df = load_flickr_annotations(csv_path=csv_path, split="val")

    train_df = parse_captions_column(train_df)
    val_df = parse_captions_column(val_df)

    train_samples = build_samples_list(train_df, images_dir)
    logging.info(f"Total train samples: {len(train_samples)}")

    val_samples = build_samples_list(val_df, images_dir)
    logging.info(f"Total val samples: {len(val_samples)}")

    train_stories = build_story_dataset(samples=train_samples)
    val_stories = build_story_dataset(samples=val_samples)

    save_clean_dataset(train_stories, save_dir, split)
    save_clean_dataset(val_stories, save_dir, split="val")
    train_dataset = StoryImageDataset(train_data_path)
    val_dataset = StoryImageDataset(val_data_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    train_dataloader = get_dataloader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    val_dataloader = get_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    optimizer = get_optimizer(model=model, lr=lr, weight_decay=weight_decay)
    scheduler = get_lr_scheduler(optimizer=optimizer)
    scaler = GradScaler()

    train_loop(
        resume=resume,
        num_epochs=num_epochs,
        latest_path=latest_path,
        best_path=best_path,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        scheduler=scheduler,
        device=device,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        resnet=resnet,
        use_amp=use_amp,
        fixed_image_path=fixed_image_path,
    )


if __name__ == "__main__":
    main()
