import os 
import pandas as pd
import ast
from typing import Dict , List, Any
import logging


def load_flickr_annotations(csv_path: str, split: str | None ="train")-> pd.DataFrame:

    if not os.path.exists(csv_path):
        logging.error(f"Invalid path {csv_path}")

    if not csv_path.endswith(".csv"):
        logging.error(f"Invalid File type {csv_path} expected .csv")

    try:    
        df = pd.read_csv(csv_path)

        if split is not None:
            df = df[df["split"] == split].reset_index(drop=True)
            return df
    
    except Exception:
        logging.exception("split is None")
        return None