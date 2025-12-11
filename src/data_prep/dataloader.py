import torch
from torch.utils.data import DataLoader
from typing import Optional, Callable


def make_collate_fn(device: Optional[torch.device] = None) -> Callable:
    """
    Returns a collate_fn that:
        -stack image tensors, input_ids, attention_mask, labels
        -optionally moves them to 'device'(if provided)
    Assumes each datasets item is a tuples:
        (image_tensor, input_ids, attention_mask, labels)
    """

    def collate_fn(batch):

        images = torch.stack([item[0] for item in batch], dim=0)
        input_ids = torch.stack([item[1] for item in batch], dim=0)
        attention_mask = torch.stack([item[2] for item in batch], dim=0)
        labels = torch.stack([item[3] for item in batch], dim=0)

        if device is not None:

            images = images.to(device=device, non_blocking=True)
            input_ids = input_ids.to(device=device, non_blocking=True)
            attention_mask = attention_mask.to(device=device, non_blocking=True)
            labels = labels.to(device=device, non_blocking=True)

        return images, input_ids, attention_mask, labels

    return collate_fn


def get_dataloader(
    dataset,
    batch_size: int = 4,
    shuffle: bool = False,
    num_workers: int = 4,
    pin_memory: bool = True,
    device: Optional[torch.device] = None,
    drop_last: bool = False,
    persistent_workers: bool = True,
) -> DataLoader:
    """
    Facotry that returns a DataLoader configured with a proper collate function.

    Recommend  usage:
        -train loader: shuffle=True, drop_last=True, batch_size = 8-16 (Depending on the GPU)
        -val/test loader: shuffle=Flase, drop_last=False, batch_size = 16-32

    Args:
        dataset: a torch.utils.data.Dataset instance that yields
                (image_tensor, input_ids, attention_mask, labels)
        device: if provided, collate_fn will batch to that device
        pin_memory: set True if using CUDA
        persistent_workers: True speeds up dataloading with multiple workers (Pytorch >=1.7)
    """
    collate_fn = make_collate_fn(device=device)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(
            pin_memory if device is not None and device.type == "cuda" else False
        ),
        drop_last=drop_last,
        collate_fn=collate_fn,
        persistent_workers=persistent_workers and num_workers > 0,
    )
