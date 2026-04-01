import torch
from config import CFG, DEVICE
from utils import apply_mixup


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, total = 0.0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()

        # Clean mixup integration
        preds, loss = apply_mixup(
            imgs,
            labels,
            criterion,
            model,
            use_mixup=True,   # you can toggle this
            alpha=0.4
        )

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            CFG["grad_clip"]
        )

        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        total += imgs.size(0)

    return total_loss / total