import torch
import logging
from torch import amp

CLIP_GRAD_VALUE = 0.1


def train_one_epoch(
    model,
    dataloader,
    optimizer: torch.optim.Optimizer,
    device,
    epoch: int,
    scheduler,
    clip_encoder,
    use_amp: bool = False,
    scaler=None,
    max_grad_norm: float = 1.0,
):

    model.to(device)
    model.train()

    running_loss = 0.0
    num_samples = 0

    for batch_idx, (images, input_ids, attn_mask, labels) in enumerate(dataloader):

        try:
            images = images.to(device)
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)

            labels = labels.to(device).long()

            with torch.no_grad():
                img_feats = clip_encoder(images)

            with amp.autocast(device_type="cuda",enabled=use_amp):
                loss, logits = model(img_feats, input_ids, attn_mask, labels)

            if loss is None:
                if batch_idx == 0:
                    logging.error("Model returned None for loss. Check if 'labels' are being passed to the forward pass.")
                continue

            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning(f"NaN/Inf loss detected at batch {batch_idx}, skipping")
                continue

            optimizer.zero_grad()

            if use_amp and scaler is not None:

                scaler.scale(loss).backward() 
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()

            else:

                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            current_batch_size = images.size(0)
            running_loss += loss.item() * current_batch_size

            num_samples += current_batch_size

            if batch_idx % 1200 == 0:
                logging.info(
                    f"Epoch [{epoch}] Batch [{batch_idx}/{len(dataloader)}] "
                    f"loss: {loss.item():.4f} "
                )
        except RuntimeError as e:
            logging.error(
                f"Runtime Error (e.g., OOM) encountered at batch {batch_idx}: {e}"
            )

            if "out of memory" in str(e):
                logging.warning("Attempting to clear cache and continue...")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            continue

    avg_loss = running_loss / max(1, num_samples)
    return avg_loss


def validate_one_epoch(
    model, dataloader, device, epoch: int, clip_encoder, use_amp: bool = False
):

    model.to(device)

    model_was_training = model.training
    model.eval()

    total_batch_loss = 0.0

    with torch.no_grad():

        for batch_idx, (images, input_ids, attn_mask, labels) in enumerate(dataloader):

            try:
                images = images.to(device)
                input_ids = input_ids.to(device)
                attn_mask = attn_mask.to(device)
                labels = labels.to(device).long()

                img_feats = clip_encoder(images)

                with amp.autocast(device_type="cuda",enabled=use_amp):
                    loss , logits = model(img_feats, input_ids, attn_mask, labels)

                total_batch_loss += loss.item()

                if batch_idx % 60 == 0:
                    logging.info(
                        f"Epoch [{epoch}] Batch [{batch_idx}/{len(dataloader)}] "
                        f"loss: {loss.item():.4f} "
                    )

            except RuntimeError as e:
                logging.error(
                    f"Runtime Error (e.g., OOM) encountered at batch {batch_idx}: {e}"
                )

                if "out of memory" in str(e):
                    logging.warning("Attempting to clear cache and continue...")
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                continue

    if model_was_training:
        model.train()

    avg_loss = total_batch_loss / len(dataloader)

    return avg_loss
