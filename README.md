# smooth-muscle-seg

Binary segmentation of smooth muscle (aSMA) in H&E histology images.

Supports multiple architectures (SMP family, UNETR-2D, MedNeXt-2D) and two training paradigms (standard supervised and OT-CFM flow matching), all driven by a single YAML config with CLI key=value overrides.

---

## Project structure

```
smooth-muscle-seg/
├── train.py                        # training entry point
├── configs/
│   ├── default.yaml                # UNet + EfficientNet-B3 (baseline)
│   ├── unetr.yaml                  # Vision Transformer encoder
│   ├── mednext.yaml                # ConvNeXt-style large-kernel UNet
│   └── flow_matching.yaml          # OT-CFM flow matching training
├── src/smooth_muscle_seg/
│   ├── model.py                    # model factory (SMP / UNETR2D / MedNeXt)
│   ├── lit_module.py               # Lightning modules (standard + flow matching)
│   ├── lit_datamodule.py           # slide-aware train/val/test split
│   ├── dataset.py                  # SegmentationDataset
│   ├── transforms.py               # albumentations augmentation pipelines
│   ├── losses.py                   # DiceBCE / Dice / Focal
│   └── metrics.py                  # Dice coefficient, IoU
└── data/                           # flat image directory (see Data section)
```

---

## Data source

H&E patches and masks are sourced from **[SegPath](https://dakomura.github.io/SegPath/)**, a large-scale pathology segmentation dataset with immunohistochemistry-derived annotations.

---

## Data format

Place H&E patches and their binary masks in a single flat directory:

```
data/
  aSMA_SmoothMuscle_{slide_id}_{x}_{y}_HE.png
  aSMA_SmoothMuscle_{slide_id}_{x}_{y}_mask.png
  ...
```

- `_HE.png` — RGB histology patch
- `_mask.png` — grayscale binary mask; pixel values `0` (background) or `1` (smooth muscle)

The datamodule splits by `slide_id` (the third `_`-separated field in the filename) to prevent patch-level leakage between train, val, and test sets.

---

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## Training

### Basic usage

```bash
# Default: UNet + EfficientNet-B3
python train.py

# Explicit config
python train.py --config configs/mednext.yaml
python train.py --config configs/unetr.yaml
python train.py --config configs/flow_matching.yaml
```

### CLI overrides

Any config key can be overridden at the command line using `section.key=value`:

```bash
python train.py training.batch_size=8 training.learning_rate=3e-4
python train.py --config configs/mednext.yaml model.variant=L model.kernel_size=5
python train.py --config configs/flow_matching.yaml flow_matching.inference_steps=50
```

### Resuming training

```bash
# Auto-find last checkpoint
python train.py --resume

# Explicit checkpoint path
python train.py --resume logs/smooth_muscle_seg/checkpoints/last.ckpt
```

---

## Architectures

### SMP family (`configs/default.yaml`)

Encoder–decoder segmentation models from [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch).

| `architecture` | Description |
|---|---|
| `unet` | U-Net with skip connections |
| `unetplusplus` | Nested U-Net++ |
| `manet` | Multi-scale Attention Net |
| `linknet` | LinkNet |
| `fpn` | Feature Pyramid Network |

Any encoder supported by SMP can be used, including Swin Transformer backbones via timm:

```yaml
model:
  architecture: unet
  encoder: timm-swin_base_patch4_window7_224  # requires: pip install timm
  encoder_weights: ~
```

### UNETR-2D (`configs/unetr.yaml`)

Vision Transformer (ViT-B/16) encoder with a 4-stage convolutional skip-connection decoder. Skip features are extracted at equal transformer-depth intervals and progressively upsampled to match the decoder resolution.

```yaml
model:
  architecture: unetr
  patch_size: 16       # image_size must be divisible by patch_size
  embed_dim: 768       # use 384 for a smaller "tiny" variant
  depth: 12
  num_heads: 12
  feature_size: 64     # base channel width for the convolutional decoder
```

### MedNeXt-2D (`configs/mednext.yaml`)

U-Net built entirely from ConvNeXt-style large-kernel depthwise blocks, adapted for 2-D histology via MONAI's `create_mednext`.

```yaml
model:
  architecture: mednext
  variant: S           # S | B | M | L  (Small → Large)
  kernel_size: 3       # depthwise kernel: 3 | 5 | 7
```

The `experiment_name` is automatically derived from these parameters: `mednext_seg_{variant}_{kernel_size}`.

---

## Training methods

### Standard supervised (`method: standard`)

Cross-entropy / Dice loss on model logits directly.

### OT-CFM flow matching (`method: flow_matching`)

Learns a velocity field `v_θ(img, x_t, t)` that transports Gaussian noise to the mask distribution via a linear conditional flow (optimal-transport CFM):

- **Training loss**: `MSE(v_θ(img, x_t, t), x_1 − x_0)` where `x_t = (1−t)·x_0 + t·x_1`
- **Inference**: Euler ODE integration from `x_0 ~ N(0,1)` over `inference_steps` steps

Any architecture can serve as the velocity backbone. See `configs/flow_matching.yaml`.

---

## Loss functions

| `loss` | Description |
|---|---|
| `dice_bce` | 0.5 · Dice + 0.5 · BCE (default) |
| `dice` | Soft Dice loss |
| `focal` | Focal loss (α=0.25, γ=2.0) |

---

## Configuration reference

All configs share the same top-level sections:

```yaml
data:
  data_dir: "data"          # flat directory of HE + mask pairs
  image_size: 512           # square crop/resize target
  num_classes: 1            # 1 = binary segmentation
  val_fraction: 0.15
  test_fraction: 0.05
  num_workers: 4

training:
  max_epochs: 100
  batch_size: 16
  learning_rate: 1.0e-4
  weight_decay: 1.0e-5
  loss: "dice_bce"          # dice_bce | dice | focal
  scheduler: "cosine"       # cosine | step | none
  early_stopping_patience: 15
  precision: "16-mixed"     # 32 | 16-mixed | bf16-mixed
  method: "standard"        # standard | flow_matching

augmentation:
  flip_p: 0.5
  rotate_limit: 90
  brightness_limit: 0.2
  contrast_limit: 0.2
  hue_shift_limit: 10
  elastic_p: 0.3
  grid_distortion_p: 0.3

logging:
  log_dir: "logs"
  experiment_name: "smooth_muscle_seg"
  save_top_k: 3
  monitor: "val/dice"
  monitor_mode: "max"
```

---

## Monitoring

Training logs and checkpoints are written to `logs/{experiment_name}/`. Launch TensorBoard with:

```bash
tensorboard --logdir logs
```

Each validation epoch logs: loss, Dice, IoU, and a 4-panel image grid (H&E | ground truth | probability map | binary prediction).

---

## GPU memory

If you hit CUDA OOM, the fastest fixes without changing the architecture:

```bash
# Reduce batch size and compensate with gradient accumulation
python train.py training.batch_size=4 training.accumulate_grad_batches=4

# Reduce fragmentation
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train.py
```

You can also reduce `data.image_size` (e.g. `384`) to cut activation memory by ~44%.
