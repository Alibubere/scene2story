import yaml
import logging
import os
from src.data_prep.flickr_loader import (
    load_flickr_annotations,
    build_samples_list,
    parse_captions_column,
)
from src.data_prep.story_generator import build_story_dataset
from src.data_prep.save_story_dataset import save_clean_dataset


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

    df = load_flickr_annotations(csv_path=csv_path, split=split)
    df = parse_captions_column(df)
    samples = build_samples_list(df, images_dir)
    logging.info(f"Total samples: {len(samples)}")
    stories = build_story_dataset(samples=samples)
    save_clean_dataset(stories,save_dir,split=split)

if __name__ == "__main__":
    main()
