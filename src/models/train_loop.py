import os
from typing import Union
import torch
import logging
from src.models.training_utils import save_checkpoint, load_checkpoint
from src.models.train import train_one_epoch, validate_one_epoch
from src.models.story_generation import generate_story_from_fixed_image


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
    processor,
    use_amp: bool,
    fixed_image_path: str = None,
):

    start_epoch = 1
    best_val_loss = 10.0
    checkpoint_to_load = None
    patience_counter = 0
    early_stop_patience = 5

    if resume:
        latest_exists = os.path.exists(latest_path)
        best_exists = os.path.exists(best_path)

        if latest_exists and best_exists:
            # Load metadata (best_val_loss) for comparison
            latest_ckpt_data = torch.load(latest_path, map_location=device)
            best_ckpt_data = torch.load(best_path, map_location=device)

            # The 'best_val_loss' key in each checkpoint reflects the best loss
            # achieved by the model saved in that specific file.
            latest_loss = latest_ckpt_data.get("best_val_loss", float("inf"))
            best_loss = best_ckpt_data.get("best_val_loss", float("inf"))

            # Compare and select the one with the lower loss
            if latest_loss <= best_loss:
                checkpoint_to_load = latest_path
                path_type = "LATEST (Better Performance)"
                best_val_loss = latest_loss
            else:
                checkpoint_to_load = best_path
                path_type = "BEST (Better Performance)"
                best_val_loss = best_loss

        elif latest_exists:
            # Only latest exists, load it
            checkpoint_to_load = latest_path
            path_type = "LATEST (Only Available)"
            # Load metadata to correctly set best_val_loss tracker
            ckpt_data = torch.load(latest_path, map_location="cpu")
            best_val_loss = ckpt_data.get("best_val_loss", best_val_loss)

        elif best_exists:
            # Only best exists, load it
            checkpoint_to_load = best_path
            path_type = "BEST (Only Available)"
            # Load metadata to correctly set best_val_loss tracker
            ckpt_data = torch.load(best_path, map_location="cpu")
            best_val_loss = ckpt_data.get("best_val_loss", best_val_loss)

        if checkpoint_to_load:
            try:
                # Load the selected checkpoint completely using your utility function
                checkpoint = load_checkpoint(
                    model, optimizer, checkpoint_to_load, scheduler, scaler, device
                )

                # Note: start_epoch is set to the *next* epoch
                start_epoch = checkpoint.get("epoch", 0) + 1
                model = checkpoint["model"]
                optimizer = checkpoint["optimizer"]
                scheduler = checkpoint["scheduler"]
                scaler = checkpoint["scaler"]

                loaded_best_val_loss = checkpoint.get("best_val_loss")
                if (
                    loaded_best_val_loss is not None
                    and loaded_best_val_loss < best_val_loss
                ):
                    best_val_loss = loaded_best_val_loss

                logging.info(
                    f"Checkpoint loaded from {checkpoint_to_load} ({path_type}). Resuming training from epoch {start_epoch}, best loss tracked: {best_val_loss:.4f}"
                )
            except RuntimeError as e:
                if "state_dict" in str(e):
                    logging.warning(
                        f"Model architecture mismatch. Starting from scratch. Error: {e}"
                    )
                    start_epoch = 1
                    best_val_loss = 10.0
                else:
                    raise

        else:
            logging.info(
                f"Resume enabled but no valid checkpoint found. Starting from scratch."
            )

    else:
        logging.info(f"Resume disabled. Starting from scratch.")

    # --- End of Checkpoint Selection Logic ---

    for epoch in range(start_epoch, num_epochs + 1):

        train_avg_loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            dataloader=train_dataloader,
            processor=processor,
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
            processor=processor,
            use_amp=use_amp,
        )

        # Step scheduler with validation loss
        if scheduler:
            scheduler.step(val_avg_loss)

        logging.info(
            f"Epoch: [{epoch}/{num_epochs}]"
            f"Train loss: {train_avg_loss:.4f} | Val loss: {val_avg_loss:.4f}"
        )

        # Generate story from fixed image for monitoring
        if fixed_image_path and os.path.exists(fixed_image_path):
            try:
                logging.info(f"--- Generating Sample Story for Epoch {epoch} ---")
                sample_story = generate_story_from_fixed_image(
                    model, processor, device, fixed_image_path, prompt="A story about"
                )
                logging.info(f"Generated Story: {sample_story}")
            except Exception as e:
                logging.warning(f"Failed to generate story: {e}")

        if val_avg_loss < best_val_loss:
            logging.info(
                f"Validation loss improved from {best_val_loss:.4f} to {val_avg_loss:.4f}. Saving Best model."
            )
            best_val_loss = val_avg_loss
            patience_counter = 0
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
            patience_counter += 1
            logging.info(
                f"Validation loss did not improve. Current best loss: {best_val_loss:.4f} (Patience: {patience_counter}/{early_stop_patience})"
            )

            # Manual LR reduction if scheduler hasn't reduced it enough
            if patience_counter == 3:
                current_lr = optimizer.param_groups[0]["lr"]
                new_lr = current_lr * 0.1
                for param_group in optimizer.param_groups:
                    param_group["lr"] = new_lr
                logging.info(
                    f"Manually reducing LR from {current_lr:.2e} to {new_lr:.2e}"
                )

            # Early stopping
            if patience_counter >= early_stop_patience:
                logging.info(
                    f"Early stopping triggered after {patience_counter} epochs without improvement"
                )
                break

        save_checkpoint(
            path=latest_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            scaler=scaler,
            best_val_loss=best_val_loss,
        )
