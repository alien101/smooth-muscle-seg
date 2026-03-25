"""Segmentation metrics (Dice, IoU) as plain functions."""

import torch


def dice_coefficient(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1.0,
    from_logits: bool = True,
) -> torch.Tensor:
    probs = torch.sigmoid(preds) if from_logits else preds
    p = (probs > threshold).float().view(-1)
    t = targets.view(-1)
    inter = (p * t).sum()
    return (2.0 * inter + smooth) / (p.sum() + t.sum() + smooth)


def iou_score(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1.0,
    from_logits: bool = True,
) -> torch.Tensor:
    probs = torch.sigmoid(preds) if from_logits else preds
    p = (probs > threshold).float().view(-1)
    t = targets.view(-1)
    inter = (p * t).sum()
    union = p.sum() + t.sum() - inter
    return (inter + smooth) / (union + smooth)
