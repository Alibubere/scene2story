import torch
import logging

def train_one_epoch(
        model,
        dataloader,
        optimizer: torch.optim.Optimizer,
        device,
        epoch:int,
        scheduler
):
    
    model.to(device)
    model.train()

    