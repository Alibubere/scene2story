from typing import List
import json
import os
import logging

def save_clean_dataset(stories:List,save_dir: str,split:str):
    """
    Save a clean dataset as a JSON Lines (.jsonl) file.

    Args:
        stories (List[Dict[str, Any]]): List of story objects to save.
        save_dir (str): Directory where the file will be saved.
        split (str): Dataset split name (e.g., 'train', 'test', 'val').
    """
    os.makedirs(save_dir,exist_ok=True)
    file_name = f"stories_{split}.jsonl"
    full_path = os.path.join(save_dir,file_name)

    with open(full_path,"w")as f:
        for story in stories:
            f.write(json.dumps(story))
            f.write("\n")

    logging.info(f"Cleaned data saved to {full_path}")