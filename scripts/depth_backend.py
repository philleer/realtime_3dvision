#!/usr/bin/env python3
"""Unified Depth Anything V2 / V3 inference backend.

V2: HuggingFace Transformers (drop-in for old V1 HF API).
V3: official ``depth_anything_3`` package (optional dependency).

Both return a float32 HxW map that is typically disparity/relative-like
(larger often means closer for DA2; always re-check with metricize_with_gt).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

# Defaults
DEFAULT_V2_HF = "depth-anything/Depth-Anything-V2-Base-hf"
DEFAULT_V3_HF = "depth-anything/da3-base"

# Legacy V1 id kept for reference / rollback
LEGACY_V1_HF = "LiheYoung/depth-anything-base-hf"


@dataclass
class DepthBackend:
    version: str  # "v2" | "v3"
    weights: str
    device: torch.device
    model: Any
    processor: Any | None = None

    @torch.inference_mode()
    def predict(self, bgr: np.ndarray) -> np.ndarray:
        """Return float32 depth/disparity map at original HxW."""
        if self.version in {"v1", "v2"}:
            return self._predict_v2(bgr)
        if self.version == "v3":
            return self._predict_v3(bgr)
        raise ValueError(f"unsupported depth version: {self.version}")

    def _predict_v2(self, bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=Image.fromarray(rgb), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        pred = self.model(**inputs).predicted_depth
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False
        )
        return pred.squeeze().float().cpu().numpy()

    def _predict_v3(self, bgr: np.ndarray) -> np.ndarray:
        # Official API accepts numpy RGB / paths / PIL; returns .depth [N,H,W]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pred = self.model.inference([rgb], export_dir=None)
        depth = pred.depth
        if isinstance(depth, torch.Tensor):
            depth = depth.detach().float().cpu().numpy()
        depth = np.asarray(depth, dtype=np.float32)
        if depth.ndim == 3:
            depth = depth[0]
        h, w = bgr.shape[:2]
        if depth.shape[0] != h or depth.shape[1] != w:
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
        return depth.astype(np.float32)


def resolve_version_and_weights(version: str | None, weights: str | None) -> tuple[str, str]:
    """Pick (version, weights) from CLI args with sensible defaults."""
    ver = (version or "v2").lower()
    if ver in {"1", "da1"}:
        ver = "v1"
    elif ver in {"2", "da2"}:
        ver = "v2"
    elif ver in {"3", "da3"}:
        ver = "v3"
    if ver not in {"v1", "v2", "v3"}:
        w_hint = weights or ""
        if "DA3" in w_hint or "da3" in w_hint:
            ver = "v3"
        elif "V2" in w_hint or "v2" in w_hint:
            ver = "v2"
        elif "LiheYoung/depth-anything" in w_hint:
            ver = "v1"
        else:
            ver = "v2"

    if weights:
        return ver, weights
    defaults = {"v1": LEGACY_V1_HF, "v2": DEFAULT_V2_HF, "v3": DEFAULT_V3_HF}
    return ver, defaults[ver]


def load_depth_backend(
    version: str | None = "v2",
    weights: str | None = None,
    device: str | torch.device = "cuda:0",
) -> DepthBackend:
    ver, w = resolve_version_and_weights(version, weights)
    if isinstance(device, str):
        if device.isdigit():
            device = torch.device(f"cuda:{device}")
        else:
            device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    if ver in {"v1", "v2"}:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        # V1/V2 HF checkpoints share AutoModelForDepthEstimation API.
        processor = AutoImageProcessor.from_pretrained(w)
        model = AutoModelForDepthEstimation.from_pretrained(w)
        model.to(device).eval()
        return DepthBackend(version=ver, weights=w, device=device, model=model, processor=processor)

    # v3
    try:
        from depth_anything_3.api import DepthAnything3
    except ImportError as e:
        raise ImportError(
            "Depth Anything V3 requires the official package. "
            "Install with: pip install 'depth-anything-3' "
            "(or clone https://github.com/ByteDance-Seed/Depth-Anything-3). "
            f"Original error: {e}"
        ) from e

    model = DepthAnything3.from_pretrained(w)
    model = model.to(device=device)
    model.eval()
    return DepthBackend(version="v3", weights=w, device=device, model=model, processor=None)


def add_depth_model_args(parser, *, default_version: str = "v2") -> None:
    """Attach common --depth-version / --depth-weights flags."""
    root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--depth-version",
        choices=["v2", "v3", "v1"],
        default=default_version,
        help="Depth Anything generation (default: v2). v1 kept for rollback.",
    )
    parser.add_argument(
        "--depth-weights",
        type=str,
        default=None,
        help=(
            "HF id or local dir. Defaults: "
            f"v2={DEFAULT_V2_HF}, v3={DEFAULT_V3_HF}, "
            f"v1={LEGACY_V1_HF}"
        ),
    )
    _ = root  # reserved for future local-default paths
