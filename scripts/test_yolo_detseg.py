#!/usr/bin/env python3
"""YOLO26-SEG detection + segmentation smoke test on demo RGB frames."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="YOLO26-SEG det/seg inference test")
    p.add_argument(
        "--weights",
        type=Path,
        default=root / "pretrained" / "yolo26m-seg.pt",
        help="Path to YOLO26-SEG weights",
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
        default=root / "tmp" / "yolo_detseg",
        help="Directory to save visualized predictions",
    )
    p.add_argument(
        "--num-frames",
        type=int,
        default=0,
        help="Max frames if source is a dir; <=0 means all",
    )
    p.add_argument("--imgsz", type=int, default=1280, help="Inference image size")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    p.add_argument("--device", default="0", help="CUDA device id or cpu")
    p.add_argument("--stride", type=int, default=1, help="Frame sampling stride in a directory")
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


def main() -> None:
    args = parse_args()
    if not args.weights.is_file():
        raise FileNotFoundError(f"weights not found: {args.weights}")

    images = collect_images(args.source, args.num_frames, args.stride)
    args.outdir.mkdir(parents=True, exist_ok=True)

    print(f"[test-detseg] weights={args.weights}")
    print(
        f"[test-detseg] device={args.device} imgsz={args.imgsz} "
        f"conf={args.conf} iou={args.iou}"
    )
    print(f"[test-detseg] frames={len(images)} -> {args.outdir}")

    import cv2

    model = YOLO(str(args.weights))
    total_boxes = 0
    total_masks = 0
    n_done = 0
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # Process one image at a time to avoid OOM on long sequences.
    for img_path in images:
        results = model.predict(
            source=str(img_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            batch=1,
            save=False,
            retina_masks=True,
            verbose=False,
        )
        r = results[0]
        vis = r.plot()  # BGR ndarray
        cv2.imwrite(str(outdir / img_path.name), vis)
        n_box = 0 if r.boxes is None else len(r.boxes)
        n_mask = 0 if r.masks is None else len(r.masks)
        total_boxes += n_box
        total_masks += n_mask
        n_done += 1
        if n_done == 1 or n_done % 100 == 0 or n_done == len(images):
            print(f"  [{n_done}/{len(images)}] {img_path.name}: boxes={n_box} masks={n_mask}")

    print(
        f"[test-detseg] done: images={n_done} "
        f"boxes={total_boxes} masks={total_masks} out={args.outdir}"
    )


if __name__ == "__main__":
    main()
