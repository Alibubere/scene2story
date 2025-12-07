import yaml 
import logging
import os
from src.data_prep.flickr_loader import(
    load_flickr_annotations,
    build_samples_list,
    parse_captions_column,
)

with open("configs/config","r") as f:
    config = yaml.save_load(f)

def logging_setup():

    log_dir = "logs"
    os.makedirs(log_dir,exist_ok=True)

    filename = "Pipeline.log"

    full_path = os.path.join(log_dir,filename)

    logging.basicConfig(
        level=logging.INFO,
        format= "%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(full_path),logging.StreamHandler()],
    )
    logging.info("Logging initialize successfully")