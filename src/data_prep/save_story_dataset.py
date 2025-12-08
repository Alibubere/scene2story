from typing import List
import json
import os
import logging

def save_clean_dataset(stories:List,save_dir: str,split:str):

    os.makedirs(save_dir,exist_ok=True)
    file_name = f"stories_{split}.jsonl"
    full_path = os.path.join(save_dir,file_name)

    with open(full_path,"w")as f:
        for story in stories:
            f.write(json.dumps(story))
            f.write("\n")

    logging.info(f"Cleaned data saved to {full_path}")