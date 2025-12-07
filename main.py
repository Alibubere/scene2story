import yaml
import logging
import os
from src.data_prep.flickr_loader import (
    load_flickr_annotations,
    build_samples_list,
    parse_captions_column,
)

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)


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

    # paths config
    paths = config["paths"]
    csv_path = paths["annotations_csv"]
    images_dir = paths["images_dir"]

    # data config
    data = config["data"]
    split = data["use_split"]
    num_preview = data["num_preview"]

    df = load_flickr_annotations(csv_path=csv_path,split=split)
    df = parse_captions_column(df)
    samples = build_samples_list(df,images_dir)
    logging.info(f"Total samples: {len(samples)}")
    for i in range(min(num_preview, len(samples))):
        s = samples[i]
        print("-" * 40)
        print("Image path:", s["image_path"])
        print("Captions:")
        for c in s["captions"]:
            print("  -", c)


if __name__ == "__main__":
    main()