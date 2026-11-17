#!/usr/bin/env python3
"""Depth Anything (ViT-B/14) monocular depth smoke test on demo RGB frames."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


# HF equivalent of official depth_anything_vitb14.pth
DEFAULT_HF_ID = "LiheYoung/depth-anything-base-hf"


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="Depth Anything depth estimation test")
    p.add_argument(
        "--weights",
        type=str,
        default=DEFAULT_HF_ID,
        help="HF model id/dir (ViT-B). Local depth_anything_vitb14.pth is original-repo format; use HF base.",
    )
    p.add_argument(
        "--source",
        type=Path,
        default=root / "data" / "jj_salon_room" / "rgb",
        help="Image file or directory of RGB frames",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=root / "tmp" / "depth_anything",
        help="Directory for RGB|depth visualization panels",
    )
    p.add_argument(
        "--num-frames",
        type=int,
        default=8,
        help="Max frames if source is a dir; <=0 means all",
    )
    p.add_argument("--stride", type=int, default=50, help="Frame sampling stride")
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--save-raw",
        action="store_true",
        help="Also save float16 .npy relative depth maps",
    )
    return p.parse_args()


def collect_images(source: Path, num_frames: int, stride: int) -> list[Path]:
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"source not found: {source}")
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = sorted(p for p in source.iterdir() if p.suffix.lower() in exts)
    if not files:
        raise FileNotFoundError(f"no images under {source}")
    sampled = files[:: max(stride, 1)]
    if num_frames > 0:
        sampled = sampled[:num_frames]
    return sampled


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    d = depth.astype(np.float32)
    d = d - np.nanmin(d)
    denom = float(np.nanmax(d)) + 1e-8
    norm = np.clip(d / denom, 0.0, 1.0)
    u8 = (norm * 255.0).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)


def make_panel(rgb_bgr: np.ndarray, depth_color: np.ndarray) -> np.ndarray:
    h, w = rgb_bgr.shape[:2]
    depth_resized = cv2.resize(depth_color, (w, h), interpolation=cv2.INTER_LINEAR)
    gap = np.full((h, 16, 3), 255, dtype=np.uint8)
    panel = np.concatenate([rgb_bgr, gap, depth_resized], axis=1)
    cv2.putText(panel, "RGB", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(
        panel,
        "Depth Anything",
        (w + 36, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        2,
    )
    return panel


@torch.inference_mode()
def predict_depth(
    model: AutoModelForDepthEstimation,
    processor: AutoImageProcessor,
    image_rgb: Image.Image,
    device: torch.device,
) -> np.ndarray:
    inputs = processor(images=image_rgb, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    # HF DepthAnything returns predicted_depth at model input resolution
    pred = outputs.predicted_depth
    h, w = image_rgb.size[1], image_rgb.size[0]
    pred = torch.nn.functional.interpolate(
        pred.unsqueeze(1),
        size=(h, w),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    return pred.float().cpu().numpy()


def main() -> None:
    # Prefer China HF mirror when unset (same as other repo downloads).
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    images = collect_images(args.source, args.num_frames, args.stride)
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    raw_dir = outdir / "raw"
    if args.save_raw:
        raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"[depth] weights={args.weights}")
    print(f"[depth] device={device}")
    print(f"[depth] frames={len(images)} -> {outdir}")

    processor = AutoImageProcessor.from_pretrained(args.weights)
    model = AutoModelForDepthEstimation.from_pretrained(args.weights)
    model.to(device).eval()

    for i, path in enumerate(images, 1):
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"failed to read {path}")
        rgb = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        depth = predict_depth(model, processor, rgb, device)
        panel = make_panel(bgr, colorize_depth(depth))
        cv2.imwrite(str(outdir / path.name), panel)
        if args.save_raw:
            np.save(raw_dir / f"{path.stem}.npy", depth.astype(np.float16))
        print(
            f"  [{i}/{len(images)}] {path.name}: "
            f"depth[{depth.min():.3f}, {depth.max():.3f}]"
        )

    print(f"[depth] done: images={len(images)} out={outdir}")


if __name__ == "__main__":
    main()
