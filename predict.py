#!/usr/bin/env python
"""Run inference on a directory of images and save predicted masks.

Usage:
    python predict.py --checkpoint logs/.../checkpoints/best.ckpt \
                      --images data/test_images \
                      --output predictions \
                      --threshold 0.5
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / "src"))

from smooth_muscle_seg.lit_module import FlowMatchingSegModule, SegmentationModule
from smooth_muscle_seg.transforms import val_transforms
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def predict_single(
    model,
    image_path: Path,
    transform,
    device: torch.device,
    threshold: float,
) -> np.ndarray:
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image.shape[:2]

    augmented = transform(image=image)
    tensor = augmented["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        prob = model.predict_mask(tensor).squeeze().cpu().numpy()

    mask = (prob > threshold).astype(np.uint8) * 255
    mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to .ckpt file")
    parser.add_argument("--images",     required=True, help="Directory of input images")
    parser.add_argument("--output",     default="predictions", help="Output directory")
    parser.add_argument("--config",     default="configs/default.yaml")
    parser.add_argument("--threshold",  type=float, default=0.5)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    method = cfg["training"].get("method", "standard")
    Cls    = FlowMatchingSegModule if method == "flow_matching" else SegmentationModule
    model  = Cls.load_from_checkpoint(args.checkpoint, cfg=cfg)
    model  = model.eval().to(device)

    transform   = val_transforms(image_size=cfg["data"]["image_size"])
    output_dir  = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = [
        p for p in sorted(Path(args.images).iterdir())
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.images}")

    for img_path in tqdm(image_paths, desc="Predicting"):
        mask = predict_single(model, img_path, transform, device, args.threshold)
        out_path = output_dir / f"{img_path.stem}_mask.png"
        cv2.imwrite(str(out_path), mask)

    print(f"Saved {len(image_paths)} masks to {output_dir}/")


if __name__ == "__main__":
    main()
