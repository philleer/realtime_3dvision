#!/usr/bin/env python3
"""Run YOLO26-SEG + Depth Anything on the same RGB frames and align visualizations."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from ultralytics import YOLO

DEFAULT_DEPTH_HF = "LiheYoung/depth-anything-base-hf"


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="Aligned YOLO-seg + depth visualization")
    p.add_argument("--yolo-weights", type=Path, default=root / "pretrained" / "yolo26m-seg.pt")
    p.add_argument("--depth-weights", type=str, default=DEFAULT_DEPTH_HF)
    p.add_argument("--source", type=Path, default=root / "data" / "jj_salon_room" / "rgb")
    p.add_argument("--outdir", type=Path, default=root / "tmp" / "yolo_depth")
    p.add_argument("--num-frames", type=int, default=8, help="<=0 means all frames")
    p.add_argument("--stride", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--device", default="0")
    p.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Output mp4 path (default: <outdir>/yolo_depth.mp4 when --write-video)",
    )
    p.add_argument(
        "--write-video",
        action="store_true",
        help="Write chronological MP4 with RGB|SEG|Depth panels",
    )
    p.add_argument("--fps", type=float, default=15.0, help="Output video FPS")
    p.add_argument(
        "--save-frames",
        action="store_true",
        default=None,
        help="Also save per-frame jpg panels (default: on if not writing video)",
    )
    p.add_argument("--no-save-frames", action="store_true", help="Do not save jpg panels")
    p.add_argument(
        "--save-raw-depth",
        action="store_true",
        help="Save float16 depth npy aligned to original HxW",
    )
    p.add_argument(
        "--panel-scale",
        type=float,
        default=1.0,
        help="Scale factor for output panel/video (e.g. 0.5 to shrink wide 3-panel)",
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
    u8 = (np.clip(d / denom, 0.0, 1.0) * 255.0).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)


def label_panel(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def make_aligned_panel(
    rgb_bgr: np.ndarray,
    seg_bgr: np.ndarray,
    depth: np.ndarray,
) -> np.ndarray:
    h, w = rgb_bgr.shape[:2]
    seg = seg_bgr if seg_bgr.shape[:2] == (h, w) else cv2.resize(seg_bgr, (w, h))
    depth_color = colorize_depth(depth)
    if depth_color.shape[:2] != (h, w):
        depth_color = cv2.resize(depth_color, (w, h), interpolation=cv2.INTER_LINEAR)

    gap = np.full((h, 12, 3), 255, dtype=np.uint8)
    panels = [
        label_panel(rgb_bgr, "RGB"),
        label_panel(seg, "YOLO26-SEG"),
        label_panel(depth_color, "Depth"),
    ]
    return np.concatenate([panels[0], gap, panels[1], gap, panels[2]], axis=1)


@torch.inference_mode()
def predict_depth(
    model: AutoModelForDepthEstimation,
    processor: AutoImageProcessor,
    bgr: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    h, w = bgr.shape[:2]
    rgb = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    inputs = processor(images=rgb, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    pred = model(**inputs).predicted_depth
    pred = torch.nn.functional.interpolate(
        pred.unsqueeze(1),
        size=(h, w),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    return pred.float().cpu().numpy()


def main() -> None:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    args = parse_args()
    if not args.yolo_weights.is_file():
        raise FileNotFoundError(f"yolo weights not found: {args.yolo_weights}")

    images = collect_images(args.source, args.num_frames, args.stride)
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    raw_dir = outdir / "depth_raw"
    if args.save_raw_depth:
        raw_dir.mkdir(parents=True, exist_ok=True)

    write_video = bool(args.write_video or args.video is not None)
    save_frames = (not args.no_save_frames) and (
        True if args.save_frames else (not write_video)
    )
    video_path = (args.video or (outdir / "yolo_depth.mp4")).resolve() if write_video else None

    gpu_index = str(args.device)
    depth_device = torch.device(
        f"cuda:{gpu_index}" if gpu_index.isdigit() and torch.cuda.is_available() else args.device
    )
    if not torch.cuda.is_available():
        depth_device = torch.device("cpu")

    print(f"[yolo-depth] yolo={args.yolo_weights}")
    print(f"[yolo-depth] depth={args.depth_weights}")
    print(f"[yolo-depth] device={args.device} depth_device={depth_device} imgsz={args.imgsz}")
    print(
        f"[yolo-depth] frames={len(images)} save_frames={save_frames} "
        f"video={video_path} fps={args.fps} -> {outdir}"
    )

    yolo = YOLO(str(args.yolo_weights))
    processor = AutoImageProcessor.from_pretrained(args.depth_weights)
    depth_model = AutoModelForDepthEstimation.from_pretrained(args.depth_weights)
    depth_model.to(depth_device).eval()

    writer: cv2.VideoWriter | None = None
    for i, path in enumerate(images, 1):
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"failed to read {path}")

        # 1) YOLO seg (plot already in original image scale with retina_masks)
        yres = yolo.predict(
            source=bgr,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            batch=1,
            save=False,
            retina_masks=True,
            verbose=False,
        )[0]
        seg_bgr = yres.plot()
        n_box = 0 if yres.boxes is None else len(yres.boxes)
        n_mask = 0 if yres.masks is None else len(yres.masks)

        # 2) Depth -> upsample to original HxW (same alignment canvas as YOLO plot)
        depth = predict_depth(depth_model, processor, bgr, depth_device)

        panel = make_aligned_panel(bgr, seg_bgr, depth)
        if args.panel_scale != 1.0:
            panel = cv2.resize(
                panel,
                (
                    max(2, int(panel.shape[1] * args.panel_scale) // 2 * 2),
                    max(2, int(panel.shape[0] * args.panel_scale) // 2 * 2),
                ),
                interpolation=cv2.INTER_AREA,
            )

        if save_frames:
            cv2.imwrite(str(outdir / path.name), panel)
        if args.save_raw_depth:
            np.save(raw_dir / f"{path.stem}.npy", depth.astype(np.float16))

        if write_video:
            if writer is None:
                h, w = panel.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(video_path), fourcc, args.fps, (w, h))
                if not writer.isOpened():
                    raise RuntimeError(f"failed to open video writer: {video_path}")
            writer.write(panel)

        if i == 1 or i % 50 == 0 or i == len(images):
            print(
                f"  [{i}/{len(images)}] {path.name}: "
                f"boxes={n_box} masks={n_mask} "
                f"depth[{depth.min():.2f},{depth.max():.2f}] "
                f"panel={panel.shape[1]}x{panel.shape[0]}"
            )

    if writer is not None:
        writer.release()
        print(f"[yolo-depth] video -> {video_path}")
    print(f"[yolo-depth] done: images={len(images)} out={outdir}")


if __name__ == "__main__":
    main()
