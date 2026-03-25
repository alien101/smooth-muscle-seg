"""PyTorch Lightning DataModule for histology segmentation."""

from pathlib import Path

import lightning as L
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .dataset import SegmentationDataset, collect_stems_and_slides
from .transforms import train_transforms, val_transforms


class SegmentationDataModule(L.LightningDataModule):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.data_dir      = cfg["data"]["data_dir"]
        self.image_size    = cfg["data"]["image_size"]
        self.num_classes   = cfg["data"]["num_classes"]
        self.val_fraction  = cfg["data"]["val_fraction"]
        self.test_fraction = cfg["data"]["test_fraction"]
        self.batch_size    = cfg["training"]["batch_size"]
        self.num_workers   = cfg["data"]["num_workers"]
        aug = cfg.get("augmentation", {})
        self._aug_kwargs = {
            "flip_p":            aug.get("flip_p", 0.5),
            "rotate_limit":      aug.get("rotate_limit", 90),
            "brightness_limit":  aug.get("brightness_limit", 0.2),
            "contrast_limit":    aug.get("contrast_limit", 0.2),
            "hue_shift_limit":   aug.get("hue_shift_limit", 10),
            "elastic_p":         aug.get("elastic_p", 0.3),
            "grid_distortion_p": aug.get("grid_distortion_p", 0.3),
        }

    def setup(self, stage: str | None = None):
        stems, slide_ids = collect_stems_and_slides(self.data_dir)

        # Split by slide ID to prevent patch-level leakage across splits
        unique_slides = sorted(set(slide_ids))
        train_val_slides, test_slides = train_test_split(
            unique_slides, test_size=self.test_fraction, random_state=42
        )
        val_size = self.val_fraction / (1.0 - self.test_fraction)
        train_slides, val_slides = train_test_split(
            train_val_slides, test_size=val_size, random_state=42
        )

        train_set = set(train_slides)
        val_set   = set(val_slides)
        test_set  = set(test_slides)

        train_stems = [s for s, sid in zip(stems, slide_ids) if sid in train_set]
        val_stems   = [s for s, sid in zip(stems, slide_ids) if sid in val_set]
        test_stems  = [s for s, sid in zip(stems, slide_ids) if sid in test_set]

        t_tf = train_transforms(image_size=self.image_size, **self._aug_kwargs)
        v_tf = val_transforms(image_size=self.image_size)

        self.train_ds = SegmentationDataset(
            self.data_dir, train_stems, transform=t_tf, num_classes=self.num_classes
        )
        self.val_ds = SegmentationDataset(
            self.data_dir, val_stems, transform=v_tf, num_classes=self.num_classes
        )
        self.test_ds = SegmentationDataset(
            self.data_dir, test_stems, transform=v_tf, num_classes=self.num_classes
        )

        print(
            f"Slides  — train: {len(train_slides)}, val: {len(val_slides)}, test: {len(test_slides)}"
        )
        print(
            f"Patches — train: {len(self.train_ds)}, val: {len(self.val_ds)}, test: {len(self.test_ds)}"
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
