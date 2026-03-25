"""Albumentations pipelines for H&E histology segmentation."""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def train_transforms(
    image_size: int = 512,
    flip_p: float = 0.5,
    rotate_limit: int = 90,
    brightness_limit: float = 0.2,
    contrast_limit: float = 0.2,
    hue_shift_limit: int = 10,
    elastic_p: float = 0.3,
    grid_distortion_p: float = 0.3,
) -> A.Compose:
    return A.Compose([
        A.RandomResizedCrop(size=(image_size, image_size), scale=(0.7, 1.0)),
        A.HorizontalFlip(p=flip_p),
        A.VerticalFlip(p=flip_p),
        A.RandomRotate90(p=flip_p),
        A.Affine(
            translate_percent=(-0.05, 0.05),
            scale=(0.9, 1.1),
            rotate=(-rotate_limit, rotate_limit),
            p=0.5,
        ),
        # H&E stain variation
        A.ColorJitter(
            brightness=brightness_limit,
            contrast=contrast_limit,
            saturation=0.1,
            hue=hue_shift_limit / 360.0,
            p=0.7,
        ),
        # Morphological distortions common in histology
        A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=elastic_p),
        A.GridDistortion(p=grid_distortion_p),
        A.GaussNoise(std_range=(0.04, 0.22), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def val_transforms(image_size: int = 512) -> A.Compose:
    return A.Compose([
        A.Resize(height=image_size, width=image_size),  # Resize still uses height/width in A 2.x
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
