import torch
import logging
from torch.cuda.amp import autocast

CLIP_GRAD_VALUE = 0.1


def train_one_epoch(
    model,
    dataloader,
    optimizer: torch.optim.Optimizer,
    device,
    epoch: int,
    scheduler,
    resnet,
    use_amp: bool = False,
    scaler=None,
):

    model.to(device)
    model.train()

    running_loss = 0.0
    num_samples = 0

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)

    for batch_idx, (images, input_ids, attn_mask, labels) in enumerate(dataloader):

        try:
            images = images.to(device)
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)

            labels = labels.to(device).long()

            with torch.no_grad():
                img_feats = resnet(images)

            with autocast(enabled=(device.type == "cuda" and use_amp)):

                logits = model(img_feats, input_ids, attn_mask)

                B, L, vocab_size = logits.shape

                flattened_logits = logits.view(-1, vocab_size)

                flattened_labels = labels.view(-1)

                loss = loss_fn(flattened_logits, flattened_labels)
                
                # Check for NaN/inf and gradients
                if torch.isnan(loss) or torch.isinf(loss):
                    logging.warning(f"NaN/Inf loss detected at batch {batch_idx}, skipping")
                    continue
                    
                # Check for extreme loss values
                if loss.item() > 20.0:
                    logging.warning(f"Extreme loss {loss.item():.4f} at batch {batch_idx}, skipping")
                    continue

            optimizer.zero_grad()

            if use_amp and scaler is not None:

                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_VALUE)

                scaler.step(optimizer)
                scaler.update()

            else:

                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_VALUE)
                optimizer.step()

            current_batch_size = images.size(0)
            running_loss += loss.item() * current_batch_size

            num_samples += current_batch_size

            if batch_idx % 1025 == 0:
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
    model, dataloader, device, epoch: int, resnet, use_amp: bool = False
):

    model.to(device)

    model_was_training = model.training
    model.eval()

    total_batch_loss = 0.0

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)

    with torch.inference_mode():

        for batch_idx, (images, input_ids, attn_mask, labels) in enumerate(dataloader):

            try:
                images = images.to(device)
                input_ids = input_ids.to(device)
                attn_mask = attn_mask.to(device)
                labels = labels.to(device).long()

                img_feats = resnet(images)

                with autocast(enabled=(device.type == "cuda" and use_amp)):
                    logits = model(img_feats, input_ids, attn_mask)

                    B, L, vocab_size = logits.shape

                    flattened_logits = logits.view(-1, vocab_size)
                    flattened_labels = labels.view(-1)

                    loss = loss_fn(flattened_logits, flattened_labels)

                total_batch_loss += loss.item()

                if batch_idx % 10 == 0:
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
