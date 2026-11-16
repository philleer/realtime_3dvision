#!/usr/bin/env python3
"""DETR-R50-panoptic segmentation smoke test (DETR-Seg smallest size)."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import DetrForSegmentation, DetrImageProcessor


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="DETR-Seg (R50-panoptic) inference test")
    p.add_argument(
        "--weights",
        type=str,
        default=str(root / "pretrained" / "detr-resnet-50-panoptic"),
        help="Local HF model dir, or hub id facebook/detr-resnet-50-panoptic",
    )
    p.add_argument(
        "--source",
        type=Path,
        default=root / "data" / "jj_salon_room" / "rgb",
        help="Image file or directory",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=root / "tmp" / "detr_seg",
        help="Visualization output directory",
    )
    p.add_argument("--num-frames", type=int, default=8, help="<=0 means all")
    p.add_argument("--stride", type=int, default=50, help="Sampling stride")
    p.add_argument("--conf", type=float, default=0.25, help="Score threshold")
    p.add_argument("--iou", type=float, default=0.5, help="Mask IoU merge threshold")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def collect_images(source: Path, num_frames: int, stride: int) -> list[Path]:
    if source.is_file():
        return [source]
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = sorted(p for p in source.iterdir() if p.suffix.lower() in exts)
    if not files:
        raise FileNotFoundError(f"no images under {source}")
    sampled = files[:: max(stride, 1)]
    if num_frames > 0:
        sampled = sampled[:num_frames]
    return sampled


def resolve_weights(weights: str, root: Path) -> str:
    local = Path(weights)
    if local.is_dir() and (local / "config.json").is_file():
        return str(local)
    # Fallback: hub id or incomplete local -> hub
    hub = "facebook/detr-resnet-50-panoptic"
    if local.name in {"detr-r50-panoptic.pth", "pytorch_model.bin"}:
        return hub
    if local.exists() and local.is_file():
        # single checkpoint file is not enough for HF API; use hub + optional cache
        return hub
    if weights.startswith("facebook/") or "/" in weights and not local.exists():
        return weights
    return hub if not local.exists() else str(local)


def palette(n: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    colors = rng.integers(64, 255, size=(max(n, 1), 3), dtype=np.uint8)
    return colors


@torch.inference_mode()
def predict_one(
    processor: DetrImageProcessor,
    model: DetrForSegmentation,
    image_bgr: np.ndarray,
    conf: float,
    iou: float,
    device: str,
) -> tuple[np.ndarray, int]:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(image_rgb)
    inputs = processor(images=pil, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)

    # Post-process to panoptic segmentation
    result = processor.post_process_panoptic_segmentation(
        outputs,
        threshold=conf,
        mask_threshold=0.5,
        overlap_mask_area_threshold=iou,
        target_sizes=[(pil.height, pil.width)],
    )[0]

    seg = result["segmentation"].detach().cpu().numpy().astype(np.int32)
    segments = result["segments_info"]
    vis = image_bgr.copy()
    colors = palette(len(segments) + 8)
    overlay = vis.copy()
    for i, info in enumerate(segments):
        sid = int(info["id"])
        mask = seg == sid
        color = colors[i % len(colors)].tolist()
        overlay[mask] = (
            0.45 * overlay[mask].astype(np.float32) + 0.55 * np.array(color, np.float32)
        ).astype(np.uint8)
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        label = f'{info.get("label_id", sid)}:{info.get("score", 0):.2f}'
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            overlay, label, (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )
    return overlay, len(segments)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    images = collect_images(args.source, args.num_frames, args.stride)
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    weights = resolve_weights(args.weights, root)
    print(f"[detr-seg] weights={weights}")
    print(f"[detr-seg] device={args.device} conf={args.conf} iou={args.iou}")
    print(f"[detr-seg] frames={len(images)} -> {outdir}")

    processor = DetrImageProcessor.from_pretrained(weights)
    model = DetrForSegmentation.from_pretrained(weights)
    model.to(args.device).eval()

    total = 0
    for i, path in enumerate(images, start=1):
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"  skip {path.name}")
            continue
        vis, n = predict_one(processor, model, bgr, args.conf, args.iou, args.device)
        cv2.imwrite(str(outdir / path.name), vis)
        total += n
        print(f"  [{i}/{len(images)}] {path.name}: segments={n}")

    print(f"[detr-seg] done: images={len(images)} segments={total} out={outdir}")


if __name__ == "__main__":
    main()
