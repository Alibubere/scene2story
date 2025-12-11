# src/data_prep/dataloader.py
import torch
from torch.utils.data import DataLoader


def collate_story_batch(batch):
    """
    Expects each item in batch to be:
      (image_tensor, input_ids, attention_mask, labels)
    Returns:
      images:        (B, 3, H, W)
      input_ids:     (B, L)
      attention_mask:(B, L)
      labels:        (B, L)
    """
    images = torch.stack([item[0] for item in batch], dim=0)
    input_ids = torch.stack([item[1] for item in batch], dim=0)
    attention_mask = torch.stack([item[2] for item in batch], dim=0)
    labels = torch.stack([item[3] for item in batch], dim=0)

    return images, input_ids, attention_mask, labels


def get_dataloader(
    dataset,
    batch_size: int = 4,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
):
    """
    Simple dataloader factory. No device logic here.
    Move tensors to device inside the training loop.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_story_batch,
    )
