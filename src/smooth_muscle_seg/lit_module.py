"""PyTorch Lightning modules for segmentation training.

SegmentationModule    — standard supervised training with any build_model architecture.
FlowMatchingSegModule — OT-CFM (linear flow matching) training; learns a velocity
                        field v_θ(img, x_t, t) that flows Gaussian noise → mask.
"""

import torch
import torch.nn.functional as F
import lightning as L
from torchvision.utils import make_grid

from .model import build_model
from .losses import build_loss
from .metrics import dice_coefficient, iou_score


# ── Validation image logging ───────────────────────────────────────────────────

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)

def _log_val_images(
    module: L.LightningModule,
    images: torch.Tensor,   # (B, 3, H, W) normalised
    masks: torch.Tensor,    # (B, 1, H, W) float {0, 1}
    preds: torch.Tensor,    # (B, 1, H, W) sigmoid probs [0, 1]
    n: int = 4,
) -> None:
    """Write a 4-row grid to TensorBoard: HE | GT | prob map | binary pred."""
    if module.logger is None:
        return
    n = min(n, images.shape[0])

    # work in float32 on CPU to avoid any fp16 display artefacts
    images = images[:n].float().cpu()
    masks  = masks[:n].float().cpu()
    preds  = preds[:n].float().cpu()

    mean = images.new_tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
    std  = images.new_tensor(_IMAGENET_STD).view(1, 3, 1, 1)
    imgs  = (images * std + mean).clamp(0, 1)           # row 1: HE image
    gt    = masks.expand(-1, 3, -1, -1)                 # row 2: GT mask  {0,1}
    prmap = preds.expand(-1, 3, -1, -1)                 # row 3: prob map [0,1]
    binary = (preds > 0.5).float().expand(-1, 3, -1, -1)  # row 4: binary pred

    grid = make_grid(torch.cat([imgs, gt, prmap, binary]), nrow=n, pad_value=0.5)
    module.logger.experiment.add_image("val/images", grid, module.current_epoch)


# ── Shared optimizer helper ────────────────────────────────────────────────────

def _make_optimizers(module: L.LightningModule, cfg: dict):
    opt = torch.optim.AdamW(
        module.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    name = cfg["training"].get("scheduler", "cosine")
    if name == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=cfg["training"]["max_epochs"]
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
    if name == "step":
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.1)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
    return opt


def _build_model_from_cfg(cfg: dict, in_channels: int, encoder_weights: str | None) -> torch.nn.Module:
    """Extract model kwargs from cfg and call build_model."""
    skip = {"architecture", "encoder", "encoder_weights", "in_channels"}
    extra = {k: v for k, v in cfg["model"].items() if k not in skip}
    return build_model(
        architecture=cfg["model"]["architecture"],
        encoder=cfg["model"].get("encoder", "efficientnet-b3"),
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        num_classes=cfg["data"]["num_classes"],
        img_size=cfg["data"]["image_size"],
        **extra,
    )


# ── Standard segmentation ─────────────────────────────────────────────────────

class SegmentationModule(L.LightningModule):
    def __init__(self, cfg: dict):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        self.model   = _build_model_from_cfg(
            cfg,
            in_channels=cfg["model"]["in_channels"],
            encoder_weights=cfg["model"].get("encoder_weights", "imagenet"),
        )
        self.loss_fn = build_loss(cfg["training"]["loss"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def predict_mask(self, images: torch.Tensor) -> torch.Tensor:
        """Return sigmoid probabilities (for external inference)."""
        return torch.sigmoid(self(images))

    def _shared_step(self, batch: dict, stage: str) -> torch.Tensor:
        images = batch["image"]
        masks  = batch["mask"]
        if masks.ndim == 3:
            masks = masks.unsqueeze(1)
        masks = masks.float()

        logits = self(images)
        loss   = self.loss_fn(logits, masks)
        dice   = dice_coefficient(logits, masks)
        iou    = iou_score(logits, masks)

        self.log(f"{stage}/loss", loss, on_step=(stage == "train"), on_epoch=True, prog_bar=True)
        self.log(f"{stage}/dice", dice, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{stage}/iou",  iou,  on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, _): return self._shared_step(batch, "train")
    def test_step(self, batch, _):     self._shared_step(batch, "test")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")
        if batch_idx == 0:
            images = batch["image"]
            masks  = batch["mask"]
            if masks.ndim == 3:
                masks = masks.unsqueeze(1)
            with torch.no_grad():
                preds = torch.sigmoid(self(images))
            _log_val_images(self, images, masks.float(), preds)

    def configure_optimizers(self): return _make_optimizers(self, self.cfg)


# ── Flow Matching segmentation ────────────────────────────────────────────────

class FlowMatchingSegModule(L.LightningModule):
    """OT-CFM (linear conditional flow matching) for segmentation.

    Training
    --------
    Given a binary mask x_1 and sampled noise x_0 ~ N(0, 1):
        x_t = (1 − t) · x_0 + t · x_1          linear interpolation
        v*  = x_1 − x_0                          target velocity
        loss = ||v_θ(img, x_t, t) − v*||²        MSE on velocity

    The velocity network receives  [image | x_t | t_spatial_map]  concatenated
    along the channel axis, so its in_channels = img_ch + num_classes + 1.

    Inference
    ---------
    Euler ODE integration from x_0 ~ N(0, 1) over `inference_steps` steps:
        x_{t+dt} = x_t + dt · v_θ(img, x_t, t)
    Final mask = sigmoid(x_1).

    Config extras
    -------------
    flow_matching:
      inference_steps: 20   # number of Euler steps at test/val time
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.inference_steps = cfg.get("flow_matching", {}).get("inference_steps", 20)

        num_classes = cfg["data"]["num_classes"]
        # velocity net takes: image + noisy mask + scalar-t spatial map
        in_ch = cfg["model"]["in_channels"] + num_classes + 1
        self.velocity_net = _build_model_from_cfg(cfg, in_channels=in_ch, encoder_weights=None)

    def forward(
        self,
        images: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        B, _, H, W = images.shape
        t_map = t.view(B, 1, 1, 1).expand(B, 1, H, W)
        return self.velocity_net(torch.cat([images, x_t, t_map], dim=1))

    def predict_mask(self, images: torch.Tensor) -> torch.Tensor:
        """Run full ODE integration and return sigmoid probabilities."""
        return self._sample(images)

    @torch.no_grad()
    def _sample(self, images: torch.Tensor) -> torch.Tensor:
        B, _, H, W = images.shape
        num_classes = self.cfg["data"]["num_classes"]
        x   = torch.randn(B, num_classes, H, W, device=images.device)
        dt  = 1.0 / self.inference_steps
        for step in range(self.inference_steps):
            t = torch.full((B,), step * dt, device=images.device)
            x = x + dt * self(images, x, t)
        return torch.sigmoid(x)

    def _shared_step(self, batch: dict, stage: str) -> torch.Tensor:
        images = batch["image"]
        masks  = batch["mask"]
        if masks.ndim == 3:
            masks = masks.unsqueeze(1)
        masks = masks.float()

        B      = images.shape[0]
        t      = torch.rand(B, device=images.device)
        x0     = torch.randn_like(masks)
        x1     = masks
        t_view = t.view(B, 1, 1, 1)
        x_t    = (1 - t_view) * x0 + t_view * x1
        loss   = F.mse_loss(self(images, x_t, t), x1 - x0)

        self.log(f"{stage}/loss", loss, on_step=(stage == "train"), on_epoch=True, prog_bar=True)
        if stage in ("val", "test"):
            probs = self._sample(images)
            self.log(f"{stage}/dice", dice_coefficient(probs, masks, from_logits=False),
                     on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"{stage}/iou",  iou_score(probs, masks, from_logits=False),
                     on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, _): return self._shared_step(batch, "train")
    def test_step(self, batch, _):     self._shared_step(batch, "test")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")
        if batch_idx == 0:
            images = batch["image"]
            masks  = batch["mask"]
            if masks.ndim == 3:
                masks = masks.unsqueeze(1)
            preds = self._sample(images)
            _log_val_images(self, images, masks.float(), preds)

    def configure_optimizers(self): return _make_optimizers(self, self.cfg)
