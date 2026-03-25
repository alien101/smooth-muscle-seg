"""Dataset for H&E histology segmentation.

Expected directory layout (flat, single folder):
    data_dir/
        {base}_{slide_id}_{x}_{y}_HE.png
        {base}_{slide_id}_{x}_{y}_mask.png
        ...

Mask values are 0 (background) and 1 (smooth muscle).
HE images are loaded as RGB (alpha channel dropped by OpenCV).
"""

from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        stems: list[str],
        transform=None,
        num_classes: int = 1,
    ):
        self.data_dir = Path(data_dir)
        self.stems = stems
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> dict:
        stem = self.stems[idx]
        image = cv2.imread(str(self.data_dir / f"{stem}_HE.png"))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(self.data_dir / f"{stem}_mask.png"), cv2.IMREAD_GRAYSCALE)
        if self.num_classes == 1:
            mask = mask.astype(np.float32)   # already 0.0 / 1.0
        else:
            mask = mask.astype(np.int64)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask  = augmented["mask"]

        return {"image": image, "mask": mask, "stem": stem}


def collect_stems_and_slides(data_dir: str | Path) -> tuple[list[str], list[str]]:
    """Scan data_dir for *_HE.png files and return (stems, slide_ids).

    stem     — filename without the _HE.png suffix, e.g.
               'aSMA_SmoothMuscle_016_008192_061440'
    slide_id — third underscore-separated field, e.g. '016'
    """
    data_dir = Path(data_dir)
    he_files = sorted(data_dir.glob("*_HE.png"))
    if not he_files:
        raise ValueError(f"No *_HE.png files found in {data_dir}")

    stems, slide_ids = [], []
    for f in he_files:
        stem = f.stem.removesuffix("_HE")          # e.g. 'aSMA_SmoothMuscle_016_...'
        slide_id = stem.split("_")[2]              # e.g. '016'
        stems.append(stem)
        slide_ids.append(slide_id)

    return stems, slide_ids
