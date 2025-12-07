import os 
import pandas as pd
import ast
from typing import Dict , List, Any
import logging


def load_flickr_annotations(csv_path: str, split: str | None ="train")-> pd.DataFrame:

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
    
    valid_splits = {"train","test","val"}

    if split not in valid_splits:
        logging.error(f"Invalid split {split}, Expected one of {valid_splits} or None.")
        raise ValueError(f"Invalid split {split}, Expected one of {valid_splits} or None.")
    
    df = df[df["split"] == split].reset_index(drop=True)
    logging.info(f"Loaded {len(df)} rows for split='{split}'.")
    return df

    

def parse_captions_column(df: pd.DataFrame) -> pd.DataFrame:

    if df is None:
        raise ValueError("DataFrame is None in parse_captions_column")
    
    df = df.copy()

    def _parse_raw(value:str):

        try:
            parsed = ast.literal_eval(value)
            
            if not isinstance(parsed,list):
                raise ValueError("Parsed 'raw' is not a list")
            return parsed
        
        except Exception:
            logging.exception("Failed to parse 'raw' value")
            raise

    df["captions"] = df["raw"].apply(_parse_raw)
    
    return df
         