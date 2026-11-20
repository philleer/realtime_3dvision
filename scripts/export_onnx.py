#!/usr/bin/env python3
"""Export YOLO26-SEG and Depth Anything Base to ONNX under pretrained/."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="Export detseg + depth models to ONNX")
    p.add_argument("--yolo-weights", type=Path, default=root / "pretrained" / "yolo26m-seg.pt")
    p.add_argument("--yolo-imgsz", type=int, nargs="+", default=[736, 1280],
                        help="YOLO export size: one int (square) or H W")
    p.add_argument(
        "--depth-weights",
        type=str,
        default="depth-anything/Depth-Anything-V2-Base-hf",
        help="HF Depth Anything V2 id (V3 is not ONNX-exported here)",
    )
    p.add_argument("--depth-size", type=int, default=518, help="Depth Anything input HxW (square)")
    p.add_argument("--depth-tag", type=str, default="v2_base", help="Output filename tag")
    p.add_argument("--outdir", type=Path, default=root / "pretrained")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--skip-yolo", action="store_true")
    p.add_argument("--skip-depth", action="store_true")
    return p.parse_args()


def export_yolo(weights: Path, outdir: Path, imgsz, opset: int) -> Path:
    from ultralytics import YOLO

    # normalize: [h,w] list / single int
    if isinstance(imgsz, list):
        imgsz = imgsz[0] if len(imgsz) == 1 else list(imgsz)
    tag = f"{imgsz[0]}x{imgsz[1]}" if isinstance(imgsz, list) else f"imgsz{imgsz}"

    outdir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))
    exported = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        simplify=True,
        dynamic=False,
        half=False,
    )
    src = Path(exported)
    dst = outdir / f"{weights.stem}_{tag}.onnx"
    if src.resolve() != dst.resolve():
        dst.write_bytes(src.read_bytes())
    print(f"[export] YOLO ONNX -> {dst} ({dst.stat().st_size / 1e6:.1f} MB)")
    return dst


def export_depth(weights: str, outdir: Path, size: int, opset: int, tag: str) -> Path:
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    outdir.mkdir(parents=True, exist_ok=True)
    processor = AutoImageProcessor.from_pretrained(weights)
    model = AutoModelForDepthEstimation.from_pretrained(weights).eval()

    class DepthWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
            return self.m(pixel_values=pixel_values).predicted_depth

    wrapped = DepthWrapper(model)
    dummy = torch.randn(1, 3, size, size, dtype=torch.float32)
    dst = outdir / f"depth_anything_{tag}_{size}.onnx"

    torch.onnx.export(
        wrapped,
        dummy,
        str(dst),
        input_names=["pixel_values"],
        output_names=["predicted_depth"],
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    meta = outdir / f"depth_anything_{tag}_{size}.json"
    meta.write_text(
        (
            "{\n"
            f'  "source": "{weights}",\n'
            f'  "tag": "{tag}",\n'
            f'  "input_size": [{size}, {size}],\n'
            f'  "opset": {opset},\n'
            f'  "mean": {getattr(processor, "image_mean", None)},\n'
            f'  "std": {getattr(processor, "image_std", None)}\n'
            "}\n"
        ),
        encoding="utf-8",
    )
    print(f"[export] Depth ONNX -> {dst} ({dst.stat().st_size / 1e6:.1f} MB)")
    return dst


def main() -> None:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    args = parse_args()
    if not args.skip_yolo:
        export_yolo(args.yolo_weights, args.outdir, args.yolo_imgsz, args.opset)
    if not args.skip_depth:
        export_depth(args.depth_weights, args.outdir, args.depth_size, args.opset, args.depth_tag)
    print("[export] done")


if __name__ == "__main__":
    main()
