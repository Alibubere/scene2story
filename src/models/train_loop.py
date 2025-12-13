import os
from typing import Union
import torch
import logging
from src.models.training_utils import save_checkpoint, load_checkpoint
from src.models.train import train_one_epoch, validate_one_epoch
from src.models.story_generation import generate_story_from_fixed_image
import math


def train_loop(
    resume: bool,
    num_epochs: int,
    latest_path: str,
    best_path: str,
    model,
    optimizer,
    scheduler,
    scaler,
    device: Union[str, torch.device],
    train_dataloader,
    val_dataloader,
    resnet,
    use_amp: bool,
    fixed_image_path: str = None,
):

    start_epoch = 1
    best_val_loss = math.inf

    checkpoint_to_load = latest_path if os.path.exists(latest_path) else None

    if checkpoint_to_load is None and os.path.exists(best_path):
        checkpoint_to_load = best_path

    if resume and checkpoint_to_load:
        checkpoint = load_checkpoint(
            model, optimizer, checkpoint_to_load, scheduler, scaler, device
        )
        start_epoch = checkpoint["epoch"]
        model = checkpoint["model"]
        optimizer = checkpoint["optimizer"]
        scheduler = checkpoint["scheduler"]
        scaler = checkpoint["scaler"]
        loaded_best_val_loss = checkpoint.get("best_val_loss")
        if loaded_best_val_loss is not None and loaded_best_val_loss < best_val_loss:
            best_val_loss = loaded_best_val_loss

        logging.info(
            f"Checkpoint loaded from {checkpoint_to_load}. Resuming training from epoch {start_epoch}, best loss tracked: {best_val_loss:.4f}"
        )

    else:
        logging.info(
            f"No valid checkpoint found or resume disabled. Starting from scratch."
        )

    for epoch in range(start_epoch, num_epochs + 1):

        train_avg_loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            dataloader=train_dataloader,
            resnet=resnet,
            device=device,
            epoch=epoch,
            scheduler=scheduler,
            scaler=scaler,
            use_amp=use_amp,
        )

        val_avg_loss = validate_one_epoch(
            model=model,
            dataloader=val_dataloader,
            device=device,
            epoch=epoch,
            resnet=resnet,
            use_amp=use_amp,
        )

        logging.info(
            f"Epoch: [{epoch}/{num_epochs}]"
            f"Train loss: {train_avg_loss:.4f} | Val loss: {val_avg_loss:.4f}"
        )
        
        # Generate story from fixed image for monitoring
        if fixed_image_path and os.path.exists(fixed_image_path):
            try:
                story = generate_story_from_fixed_image(
                    model, resnet, device, fixed_image_path
                )
                logging.info(f"Epoch {epoch} Generated Story: {story}")
            except Exception as e:
                logging.warning(f"Failed to generate story: {e}")

        if val_avg_loss < best_val_loss:
            prev_best = best_val_loss
            best_val_loss = val_avg_loss
            
            logging.info(
                f"Validation loss improved from {prev_best:.4f} to {best_val_loss:.4f}. Saving Best model."
            )

            save_checkpoint(
                path=best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                scaler=scaler,
                best_val_loss=best_val_loss,
            )

        else:
            logging.info(
                f"Validation loss did not improve. Current best loss: {best_val_loss:.4f}"
            )

        save_checkpoint(
            path=latest_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            scaler=scaler,
            best_val_loss=best_val_loss,
        )
