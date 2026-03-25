#!/usr/bin/env python
"""Training entry point.

Usage:
    python train.py                          # uses configs/default.yaml
    python train.py --config configs/my.yaml
    python train.py --config configs/default.yaml training.batch_size=8
    python train.py --resume                                         # auto-find last.ckpt
    python train.py --resume logs/smooth_muscle_seg/checkpoints/last.ckpt
"""

import argparse
import sys
from pathlib import Path

import yaml
import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger

# make sure src/ is on the path when running without installing
sys.path.insert(0, str(Path(__file__).parent / "src"))

from smooth_muscle_seg.lit_module import FlowMatchingSegModule, SegmentationModule
from smooth_muscle_seg.lit_datamodule import SegmentationDataModule


def deep_update(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def parse_overrides(overrides: list[str]) -> dict:
    """Parse 'section.key=value' CLI overrides into a nested dict."""
    result = {}
    for item in overrides:
        key_path, _, value = item.partition("=")
        keys = key_path.split(".")
        d = result
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        # try to coerce to int / float / bool
        for cast in (int, float):
            try:
                value = cast(value)
                break
            except ValueError:
                pass
        if value == "true":
            value = True
        elif value == "false":
            value = False
        d[keys[-1]] = value
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--resume", nargs="?", const="latest", default=None, metavar="CKPT",
                        help="resume training; omit path to auto-find last.ckpt")
    parser.add_argument("overrides", nargs="*", help="key=value overrides, e.g. training.batch_size=8")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.overrides:
        deep_update(cfg, parse_overrides(args.overrides))

    # Resolve template variables in experiment_name using model config values
    cfg["logging"]["experiment_name"] = cfg["logging"]["experiment_name"].format(
        **cfg.get("model", {})
    )

    L.seed_everything(42, workers=True)

    datamodule = SegmentationDataModule(cfg)
    method     = cfg["training"].get("method", "standard")
    module     = FlowMatchingSegModule(cfg) if method == "flow_matching" else SegmentationModule(cfg)

    log_cfg = cfg["logging"]
    logger = TensorBoardLogger(
        save_dir=log_cfg["log_dir"],
        name=log_cfg["experiment_name"],
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=Path(log_cfg["log_dir"]) / log_cfg["experiment_name"] / "checkpoints",
            filename="{epoch:03d}-{val/dice:.4f}",
            monitor=log_cfg["monitor"],
            mode=log_cfg["monitor_mode"],
            save_top_k=log_cfg["save_top_k"],
            save_last=True,
        ),
        EarlyStopping(
            monitor=log_cfg["monitor"],
            mode=log_cfg["monitor_mode"],
            patience=cfg["training"]["early_stopping_patience"],
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = L.Trainer(
        max_epochs=cfg["training"]["max_epochs"],
        precision=cfg["training"]["precision"],
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        deterministic=False,  # set True for full reproducibility (slower)
    )

    resume = args.resume
    if resume == "latest":
        ckpt_dir = Path(log_cfg["log_dir"]) / log_cfg["experiment_name"] / "checkpoints"
        last = ckpt_dir / "last.ckpt"
        if not last.exists():
            raise FileNotFoundError(f"No last.ckpt found in {ckpt_dir}")
        resume = str(last)
        print(f"Resuming from {resume}")

    trainer.fit(module, datamodule=datamodule, ckpt_path=resume)
    trainer.test(module, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
