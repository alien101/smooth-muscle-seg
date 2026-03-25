"""Model factory: SMP family, UNETR-2D (ViT), MedNeXt-2D (ConvNeXt), and utilities.

Architecture overview
---------------------
SMP family (segmentation_models_pytorch):
    unet | unetplusplus | manet | linknet | fpn

    Any SMP architecture also accepts timm-backed ViT/Swin encoders, e.g.:
        architecture: unet
        encoder: timm-swin_base_patch4_window7_224
    Requires `pip install timm`.

UNETR-2D  (architecture: "unetr"):
    ViT patch encoder + 4-stage convolutional skip-connection decoder.
    Skip features extracted at equal transformer-depth intervals.

MedNeXt-2D  (architecture: "mednext"):
    U-Net built entirely from ConvNeXt-style large-kernel depthwise blocks.
    Matches the MedNeXt design philosophy adapted for 2-D histology.
"""

import math
from monai.networks.nets.mednext import create_mednext
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


# ── SMP factory ───────────────────────────────────────────────────────────────

_SMP_FACTORY = {
    "unet":         smp.Unet,
    "unetplusplus": smp.UnetPlusPlus,
    "manet":        smp.MAnet,
    "linknet":      smp.Linknet,
    "fpn":          smp.FPN,
}




# ── UNETR-2D ──────────────────────────────────────────────────────────────────

class UNETR2D(nn.Module):
    """UNETR for 2-D images: ViT encoder with convolutional skip-connection decoder.

    Skip features are extracted at 4 equally-spaced transformer layers and
    progressively upsampled to the matching decoder resolution before merging.

    For img_size=512, patch_size=16 (grid G=32):
        skip0 (fine)  → 3 ups → 256×256
        skip1         → 2 ups → 128×128
        skip2         → 1 up  →  64×64
        skip3 (bot)   → 0 ups →  32×32  (bottleneck)
    Then 1 extra upsample → 512×512 → head.

    Args:
        in_channels:  Input image channels.
        num_classes:  Output channels.
        img_size:     Square spatial size of input (must be divisible by patch_size).
        patch_size:   ViT patch size.
        embed_dim:    Transformer embedding dimension.
        depth:        Number of transformer layers.
        num_heads:    Attention heads (embed_dim must be divisible by this).
        feature_size: Base channel width for the convolutional decoder.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        img_size: int = 512,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        feature_size: int = 64,
    ):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        G = img_size // patch_size  # patch-grid side length (e.g. 32 for 512/16)
        self.G = G
        F = feature_size

        # ── Patch embedding ───────────────────────────────────────────────────
        self.patch_proj = nn.Conv2d(in_channels, embed_dim, patch_size, stride=patch_size)
        self.pos_embed  = nn.Parameter(torch.zeros(1, G * G, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # ── Transformer blocks (individual, for per-layer skip extraction) ────
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads,
                dim_feedforward=embed_dim * 4, dropout=0.0,
                batch_first=True, norm_first=True,
            )
            for _ in range(depth)
        ])

        # Skip indices at depth/4 intervals (0-indexed block outputs)
        q = max(1, depth // 4)
        self.skip_at = [q - 1, 2 * q - 1, 3 * q - 1, depth - 1]

        # ── Skip projection + upsampling chains ───────────────────────────────
        # skip_chains[i] = [projection, *conv_upsamples]
        #   i=0 (finest encoder features): 3 upsamples → G×8
        #   i=1:                           2 upsamples → G×4
        #   i=2:                           1 upsample  → G×2
        #   i=3 (bottleneck):              0 upsamples → G
        def make_skip_chain(n_ups: int) -> nn.ModuleList:
            chain: nn.ModuleList = nn.ModuleList()
            chain.append(nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, F)))
            for _ in range(n_ups):
                chain.append(nn.Sequential(
                    nn.ConvTranspose2d(F, F, 2, stride=2),
                    nn.InstanceNorm2d(F), nn.GELU(),
                ))
            return chain

        self.skip_chains = nn.ModuleList([
            make_skip_chain(3),
            make_skip_chain(2),
            make_skip_chain(1),
            make_skip_chain(0),
        ])

        # ── Decoder ───────────────────────────────────────────────────────────
        self.bot_conv = nn.Sequential(
            nn.Conv2d(F, F * 2, 3, padding=1),
            nn.InstanceNorm2d(F * 2), nn.GELU(),
        )

        def up_block(in_c: int) -> nn.Sequential:
            return nn.Sequential(
                nn.ConvTranspose2d(in_c, F, 2, stride=2),
                nn.Conv2d(F, F, 3, padding=1),
                nn.InstanceNorm2d(F), nn.GELU(),
            )

        def merge_block() -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(F * 2, F, 3, padding=1),
                nn.InstanceNorm2d(F), nn.GELU(),
            )

        self.dec1_up    = up_block(F * 2)   # G   → G×2
        self.dec1_merge = merge_block()
        self.dec2_up    = up_block(F)        # G×2 → G×4
        self.dec2_merge = merge_block()
        self.dec3_up    = up_block(F)        # G×4 → G×8
        self.dec3_merge = merge_block()

        # Extra 2× upsamples to bridge G×8 → img_size when G×8 < img_size
        after_skips = 8 * G
        n_extra = max(0, round(math.log2(img_size / after_skips))) if after_skips < img_size else 0
        extra: list[nn.Module] = []
        for _ in range(n_extra):
            extra += [nn.ConvTranspose2d(F, F, 2, stride=2), nn.InstanceNorm2d(F), nn.GELU()]
        self.dec_extra = nn.Sequential(*extra) if extra else nn.Identity()

        self.head = nn.Conv2d(F, num_classes, 1)

    def _process_skip(self, tokens: torch.Tensor, chain: nn.ModuleList) -> torch.Tensor:
        """Project token sequence then upsample to target spatial resolution."""
        B = tokens.shape[0]
        x = chain[0](tokens)                              # (B, N, F)
        x = x.transpose(1, 2).reshape(B, -1, self.G, self.G)  # (B, F, G, G)
        for layer in chain[1:]:
            x = layer(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_proj(x)                              # (B, D, G, G)
        x = x.flatten(2).transpose(1, 2) + self.pos_embed  # (B, N, D)

        skips: list[torch.Tensor] = []
        skip_set = set(self.skip_at)
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in skip_set:
                skips.append(x)

        s0  = self._process_skip(skips[0], self.skip_chains[0])  # G×8
        s1  = self._process_skip(skips[1], self.skip_chains[1])  # G×4
        s2  = self._process_skip(skips[2], self.skip_chains[2])  # G×2
        bot = self._process_skip(skips[3], self.skip_chains[3])  # G

        x = self.bot_conv(bot)
        x = self.dec1_merge(torch.cat([self.dec1_up(x), s2], dim=1))
        x = self.dec2_merge(torch.cat([self.dec2_up(x), s1], dim=1))
        x = self.dec3_merge(torch.cat([self.dec3_up(x), s0], dim=1))
        x = self.dec_extra(x)
        return self.head(x)


# ── Public factory ────────────────────────────────────────────────────────────

def build_model(
    architecture: str = "unet",
    encoder: str = "efficientnet-b3",
    encoder_weights: str | None = "imagenet",
    in_channels: int = 3,
    num_classes: int = 1,
    img_size: int = 512,
    **kwargs,
) -> nn.Module:
    """Build a segmentation model from config parameters.

    SMP architectures  (pass encoder / encoder_weights as usual):
        unet | unetplusplus | manet | linknet | fpn
        Also supports timm encoders, e.g. encoder="timm-swin_base_patch4_window7_224"

    UNETR-2D  (architecture="unetr"):
        kwargs: patch_size(16), embed_dim(768), depth(12),
                num_heads(12), feature_size(64)

    MedNeXt-2D  (architecture="mednext"):
        kwargs: variant("S" | "B" | "M" | "L"), kernel_size(3),
                deep_supervision(False)
    """
    if architecture in _SMP_FACTORY:
        return _SMP_FACTORY[architecture](
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=None,
        )

    if architecture == "unetr":
        return UNETR2D(
            in_channels=in_channels,
            num_classes=num_classes,
            img_size=img_size,
            patch_size=int(kwargs.get("patch_size", 16)),
            embed_dim=int(kwargs.get("embed_dim", 768)),
            depth=int(kwargs.get("depth", 12)),
            num_heads=int(kwargs.get("num_heads", 12)),
            feature_size=int(kwargs.get("feature_size", 64)),
        )

    if architecture == "mednext":
        return create_mednext(
            variant=kwargs.get("variant", "S"),
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=num_classes,
            kernel_size=int(kwargs.get("kernel_size", 3)),
            deep_supervision=bool(kwargs.get("deep_supervision", False)),
        )

    raise ValueError(
        f"Unknown architecture '{architecture}'. "
        f"Choose from: {list(_SMP_FACTORY) + ['unetr', 'mednext']}"
    )
