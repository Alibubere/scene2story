import os
import pandas as pd
import ast
from typing import Dict, List, Any
import logging


def load_flickr_annotations(csv_path: str, split: str) -> pd.DataFrame:

    if not os.path.exists(csv_path):
        logging.error(f"CSV file not found: {csv_path}")
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    if not csv_path.endswith(".csv"):
        logging.error(f"Invalid File type for annotation: {csv_path}. Expected .csv")
        raise ValueError(f"Invalid File type for annotation: {csv_path}. Expected .csv")

    df = pd.read_csv(csv_path)

    if split is None or split == "all":
        df = df.reset_index(drop=True)
        logging.info(f"Loaded {len(df)} rows (no split filtering)")
        return df

    valid_splits = {"train", "test", "val"}

    if split not in valid_splits:
        logging.error(f"Invalid split {split}, Expected one of {valid_splits} or None.")
        raise ValueError(
            f"Invalid split {split}, Expected one of {valid_splits} or None."
        )

    df = df[df["split"] == split].reset_index(drop=True)
    logging.info(f"Loaded {len(df)} rows for split='{split}'.")
    return df


def parse_captions_column(df: pd.DataFrame) -> pd.DataFrame:

    if df is None:
        raise ValueError("DataFrame is None in parse_captions_column")

    df = df.copy()

    def _parse_raw(value: str):

        try:
            parsed = ast.literal_eval(value)

            if not isinstance(parsed, list):
                raise ValueError("Parsed 'raw' is not a list")
            return parsed

        except Exception:
            logging.exception("Failed to parse 'raw' value")
            raise

    df["captions"] = df["raw"].apply(_parse_raw)

    return df


def build_samples_list(df: pd.DataFrame, images_dir: str) -> List[Dict[str, Any]]:

    if df is None:
        raise ValueError("Dataframe is None in build_samples_list")

    if not os.path.isdir(images_dir):
        logging.error(f"Images directory not found: {images_dir}")
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    samples: List[Dict[str, Any]] = []

    for index, row in df.iterrows():

        image_path = os.path.join(images_dir, row["filename"])

        sample: Dict[str, Any] = {
            "image_path": image_path,
            "captions": row["captions"],
        }

        if "split" in df.columns:
            sample["split"] = row["split"]

        if "img_id" in df.columns:
            sample["img_id"] = row["img_id"]

        samples.append(sample)

    logging.info(f"Built {len(samples)} samples from annotaions")
    return samples