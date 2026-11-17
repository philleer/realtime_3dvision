#!/usr/bin/env python3
"""Project YOLO RGB segmentation into a top-down BEV (camera origin = world origin)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="RGB-seg -> top-down BEV (cam=world)")
    p.add_argument("--yolo-weights", type=Path, default=root / "pretrained" / "yolo26m-seg.pt")
    p.add_argument("--source", type=Path, default=root / "data" / "jj_salon_room" / "rgb")
    p.add_argument(
        "--depth-dir",
        type=Path,
        default=root / "tmp" / "yolo_depth" / "depth_raw",
        help="Predicted depth .npy (RGB-aligned) or GT depth .png dir",
    )
    p.add_argument(
        "--calib",
        type=Path,
        default=root / "data" / "jj_salon_room" / "scan-003" / "calib.json",
    )
    p.add_argument("--outdir", type=Path, default=root / "tmp" / "seg_bev")
    p.add_argument("--num-frames", type=int, default=8)
    p.add_argument("--stride", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--device", default="0")
    p.add_argument("--camera", choices=["color", "depth"], default="color")
    p.add_argument(
        "--depth-unit",
        choices=["auto", "m", "mm", "relative"],
        default="auto",
    )
    p.add_argument("--depth-scale", type=float, default=1.0)
    p.add_argument("--metricize-gt", action="store_true", default=True)
    p.add_argument("--no-metricize-gt", action="store_true")
    p.add_argument("--gt-depth-dir", type=Path, default=root / "data" / "jj_salon_room" / "depth")
    p.add_argument("--min-depth-m", type=float, default=0.05)
    p.add_argument("--max-depth-m", type=float, default=8.0)
    p.add_argument("--stride-px", type=int, default=2, help="Pixel stride for back-projection")
    p.add_argument("--mpp", type=float, default=0.02, help="BEV metres per pixel")
    p.add_argument("--x-range", type=float, nargs=2, default=(-3.0, 3.0), metavar=("XMIN", "XMAX"))
    p.add_argument("--z-range", type=float, nargs=2, default=(0.0, 6.0), metavar=("ZMIN", "ZMAX"))
    p.add_argument(
        "--y-keep",
        type=float,
        nargs=2,
        default=(-2.5, 2.5),
        metavar=("YMIN", "YMAX"),
        help="Keep camera-Y band (OpenCV: +Y down) before BEV raster",
    )
    p.add_argument("--write-video", action="store_true")
    p.add_argument("--fps", type=float, default=8.0)
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


def load_calib(path: Path, camera: str) -> tuple[float, float, float, float, int, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cam = data[camera]
    K = np.asarray(cam["K"], dtype=np.float64)
    return float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]), int(cam["width"]), int(cam["height"])


def scale_intrinsics(
    fx: float, fy: float, cx: float, cy: float, w0: int, h0: int, w: int, h: int
) -> tuple[float, float, float, float]:
    sx, sy = w / w0, h / h0
    return fx * sx, fy * sy, cx * sx, cy * sy


def load_depth(path: Path, unit: str) -> tuple[np.ndarray, str]:
    if path.suffix.lower() == ".npy":
        d = np.load(path).astype(np.float32)
        resolved = "relative" if unit == "auto" else unit
        if resolved == "mm":
            return d / 1000.0, "m"
        return d, resolved if resolved != "m" else "m"

    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise RuntimeError(f"failed to read depth: {path}")
    d = raw.astype(np.float32)
    resolved = "mm" if unit == "auto" else unit
    if resolved == "mm":
        return d / 1000.0, "m"
    return d, resolved


def resize_depth_to_rgb(depth: np.ndarray, wh: tuple[int, int]) -> np.ndarray:
    w, h = wh
    if depth.shape[0] == h and depth.shape[1] == w:
        return depth
    return cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)


def metricize_with_gt(pred_m: np.ndarray, gt_path: Path, rgb_wh: tuple[int, int]) -> tuple[np.ndarray, str]:
    """Align relative Depth-Anything output to GT metres (handles disparity-like maps)."""
    gt, _ = load_depth(gt_path, "auto")
    gt = resize_depth_to_rgb(gt, rgb_wh)
    pred = resize_depth_to_rgb(pred_m, rgb_wh)
    valid = (gt > 0.2) & (gt < 5.0) & np.isfinite(pred) & (pred > 1e-3)
    if valid.sum() < 100:
        return pred, "relative(unscaled)"
    corr = float(np.corrcoef(pred[valid], gt[valid])[0, 1])
    if corr < 0.0:
        scale = float(np.median(gt[valid] * pred[valid]))
        return scale / np.clip(pred, 1e-6, None), f"m(disp/gt,corr={corr:.2f})"
    scale = float(np.median(gt[valid] / pred[valid]))
    return pred * scale, f"m(depth*s,corr={corr:.2f})"


def find_depth(depth_dir: Path, stem: str) -> Path:
    for ext in (".npy", ".png"):
        cand = depth_dir / f"{stem}{ext}"
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"no depth for {stem} under {depth_dir}")


def paint_seg_overlay(bgr: np.ndarray, result) -> np.ndarray:
    """Instance-colored segmentation overlay (BGR), background kept from RGB."""
    overlay = bgr.copy()
    if result.masks is None or len(result.masks) == 0:
        return overlay
    masks = result.masks.data.detach().cpu().numpy()  # (N,H,W) at model/mask scale
    # Align masks to original image size
    h, w = bgr.shape[:2]
    rng = np.random.default_rng(0)
    # Prefer stable colors from boxes cls if available
    names = result.names if hasattr(result, "names") else {}
    for i in range(masks.shape[0]):
        m = masks[i]
        if m.shape[0] != h or m.shape[1] != w:
            m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        bin_m = m > 0.5
        if not bin_m.any():
            continue
        color = tuple(int(c) for c in rng.integers(40, 255, size=3))
        if result.boxes is not None and i < len(result.boxes):
            # Deterministic-ish color from class id
            cls_id = int(result.boxes.cls[i].item()) if result.boxes.cls is not None else i
            rng_c = np.random.default_rng(cls_id * 97 + 13)
            color = tuple(int(c) for c in rng_c.integers(50, 240, size=3))
            _ = names.get(cls_id, "")
        overlay[bin_m] = (
            0.35 * overlay[bin_m].astype(np.float32) + 0.65 * np.array(color, np.float32)
        ).astype(np.uint8)
    # Draw box labels via ultralytics plot for readability, then blend masks we already painted
    plotted = result.plot(masks=True, boxes=True, conf=True, labels=True)
    if plotted.shape[:2] != (h, w):
        plotted = cv2.resize(plotted, (w, h))
    return plotted


def backproject_colors(
    depth_m: np.ndarray,
    color_bgr: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    min_z: float,
    max_z: float,
    ymin: float,
    ymax: float,
    stride_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = depth_m.shape
    us = np.arange(0, w, stride_px)
    vs = np.arange(0, h, stride_px)
    uu, vv = np.meshgrid(us, vs)
    z = depth_m[vv, uu]
    mask = np.isfinite(z) & (z >= min_z) & (z <= max_z)
    uu = uu[mask].astype(np.float32)
    vv = vv[mask].astype(np.float32)
    z = z[mask].astype(np.float32)
    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy
    keep = (y >= ymin) & (y <= ymax)
    uu, vv, x, y, z = uu[keep], vv[keep], x[keep], y[keep], z[keep]
    pts = np.stack([x, y, z], axis=1)
    cols = color_bgr[vv.astype(np.int32), uu.astype(np.int32)]
    return pts, cols


def rasterize_bev(
    pts: np.ndarray,
    cols_bgr: np.ndarray,
    x_range: tuple[float, float],
    z_range: tuple[float, float],
    mpp: float,
) -> np.ndarray:
    """Top-down BEV looking along -Y: u↔X, v↔Z (Z forward up the image)."""
    xmin, xmax = x_range
    zmin, zmax = z_range
    w = max(1, int(np.ceil((xmax - xmin) / mpp)))
    h = max(1, int(np.ceil((zmax - zmin) / mpp)))
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    depth_buf = np.full((h, w), np.inf, dtype=np.float32)  # keep highest point (-Y up)

    if len(pts) == 0:
        return canvas

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    col = ((x - xmin) / mpp).astype(np.int32)
    row = ((zmax - z) / mpp).astype(np.int32)  # zmax at top of image
    valid = (col >= 0) & (col < w) & (row >= 0) & (row < h)
    col, row, y, cols_bgr = col[valid], row[valid], y[valid], cols_bgr[valid]

    # Prefer smaller camera-Y (higher in scene when Y points down)
    order = np.argsort(-y)  # farthest down first, then overwrite with higher
    col, row, y, cols_bgr = col[order], row[order], y[order], cols_bgr[order]
    for c, r, yy, bgr in zip(col, row, y, cols_bgr):
        if yy < depth_buf[r, c]:
            depth_buf[r, c] = yy
            canvas[r, c] = bgr
    return canvas


def draw_bev_overlay(bev: np.ndarray, x_range, z_range, mpp: float, title: str) -> np.ndarray:
    out = bev.copy()
    h, w = out.shape[:2]
    xmin, xmax = x_range
    zmin, zmax = z_range
    # Camera origin (0,0) in XZ
    ox = int(round((0.0 - xmin) / mpp))
    oy = int(round((zmax - 0.0) / mpp))
    if 0 <= ox < w and 0 <= oy < h:
        cv2.drawMarker(out, (ox, oy), (0, 255, 255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=18, thickness=2)
        cv2.putText(out, "cam", (ox + 8, oy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
    # Axes: +X right, +Z up (image up)
    if 0 <= ox < w and 0 <= oy < h:
        cv2.arrowedLine(out, (ox, oy), (min(w - 1, ox + 60), oy), (0, 0, 255), 2, tipLength=0.25)
        cv2.putText(out, "+X", (min(w - 40, ox + 64), oy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.arrowedLine(out, (ox, oy), (ox, max(0, oy - 60)), (0, 255, 0), 2, tipLength=0.25)
        cv2.putText(out, "+Z", (ox + 6, max(16, oy - 64)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    # Grid every 1 m
    for xv in np.arange(np.ceil(xmin), xmax + 1e-6, 1.0):
        c = int(round((xv - xmin) / mpp))
        if 0 <= c < w:
            cv2.line(out, (c, 0), (c, h - 1), (40, 40, 40), 1)
    for zv in np.arange(np.ceil(zmin), zmax + 1e-6, 1.0):
        r = int(round((zmax - zv) / mpp))
        if 0 <= r < h:
            cv2.line(out, (0, r), (w - 1, r), (40, 40, 40), 1)
    cv2.putText(out, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        out,
        f"{mpp*100:.0f}cm/px  X[{xmin:.1f},{xmax:.1f}] Z[{zmin:.1f},{zmax:.1f}] m",
        (12, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return out


def _label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, text, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def _resize_to_h(img: np.ndarray, hh: int) -> np.ndarray:
    scale = hh / img.shape[0]
    return cv2.resize(img, (max(1, int(round(img.shape[1] * scale))), hh))


def _pad_to_w(img: np.ndarray, ww: int) -> np.ndarray:
    if img.shape[1] == ww:
        return img
    if img.shape[1] > ww:
        return cv2.resize(img, (ww, img.shape[0]))
    return cv2.copyMakeBorder(img, 0, 0, 0, ww - img.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0))


def make_panel(rgb_bgr, seg_bgr, bev_rgb, bev_seg) -> np.ndarray:
    """2x2: RGB | SEG / BEV-RGB | BEV-SEG, BEV upscaled to perspective row height."""
    row_h = max(360, min(rgb_bgr.shape[0], 720) // 1)
    # Perspective row
    top = np.hstack(
        [
            _label(_resize_to_h(rgb_bgr, row_h), "RGB"),
            _label(_resize_to_h(seg_bgr, row_h), "SEG"),
        ]
    )
    # BEV cells: match each half of the top row width, keep square-ish
    half_w = top.shape[1] // 2
    cell = max(half_w, row_h)

    def fit_bev(bev: np.ndarray) -> np.ndarray:
        # Upscale BEV into a square cell then center-pad to (row_h x half_w)
        side = min(cell, max(bev.shape[0], bev.shape[1]) * 4)  # allow upscale
        side = max(side, row_h)
        scale = side / max(bev.shape[0], bev.shape[1])
        resized = cv2.resize(
            bev,
            (max(1, int(round(bev.shape[1] * scale))), max(1, int(round(bev.shape[0] * scale)))),
            interpolation=cv2.INTER_NEAREST,
        )
        # Crop/pad to row_h x half_w
        rh, rw = resized.shape[:2]
        if rw > half_w:
            x0 = (rw - half_w) // 2
            resized = resized[:, x0 : x0 + half_w]
        if rh > row_h:
            y0 = (rh - row_h) // 2
            resized = resized[y0 : y0 + row_h]
        canvas = np.zeros((row_h, half_w, 3), dtype=np.uint8)
        y0 = (row_h - resized.shape[0]) // 2
        x0 = (half_w - resized.shape[1]) // 2
        canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
        return canvas

    bot = np.hstack([fit_bev(bev_rgb), fit_bev(bev_seg)])
    bot = _pad_to_w(bot, top.shape[1])
    top = _pad_to_w(top, bot.shape[1])
    return np.vstack([top, bot])

def main() -> None:
    args = parse_args()
    if args.no_metricize_gt:
        args.metricize_gt = False

    images = collect_images(args.source, args.num_frames, args.stride)
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    fx0, fy0, cx0, cy0, w0, h0 = load_calib(args.calib, args.camera)
    x_range = (float(args.x_range[0]), float(args.x_range[1]))
    z_range = (float(args.z_range[0]), float(args.z_range[1]))
    y_keep = (float(args.y_keep[0]), float(args.y_keep[1]))

    print(f"[bev] calib={args.calib} camera={args.camera} K0=({fx0:.1f},{fy0:.1f},{cx0:.1f},{cy0:.1f})")
    print(f"[bev] frames={len(images)} mpp={args.mpp} X={x_range} Z={z_range} cam=world(I)")
    print(f"[bev] yolo={args.yolo_weights} depth={args.depth_dir} -> {outdir}")

    model = YOLO(str(args.yolo_weights))
    writer = None
    video_path = outdir / "seg_bev.mp4"

    for i, rgb_path in enumerate(images, 1):
        bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"failed to read {rgb_path}")
        h, w = bgr.shape[:2]

        depth_path = find_depth(args.depth_dir, rgb_path.stem)
        depth, unit = load_depth(depth_path, args.depth_unit)
        depth = resize_depth_to_rgb(depth, (w, h))
        if unit == "relative":
            depth = depth * float(args.depth_scale)
            if args.metricize_gt:
                gt = args.gt_depth_dir / f"{rgb_path.stem}.png"
                if gt.is_file():
                    depth, unit = metricize_with_gt(depth, gt, (w, h))
                else:
                    unit = "relative*scale"
            else:
                unit = "relative*scale"

        yres = model.predict(
            source=bgr,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
            retina_masks=True,
        )[0]
        seg_bgr = paint_seg_overlay(bgr, yres)
        n_mask = 0 if yres.masks is None else len(yres.masks)

        fx, fy, cx, cy = scale_intrinsics(fx0, fy0, cx0, cy0, w0, h0, w, h)
        pts_rgb, cols_rgb = backproject_colors(
            depth, bgr, fx, fy, cx, cy, args.min_depth_m, args.max_depth_m, y_keep[0], y_keep[1], args.stride_px
        )
        pts_seg, cols_seg = backproject_colors(
            depth, seg_bgr, fx, fy, cx, cy, args.min_depth_m, args.max_depth_m, y_keep[0], y_keep[1], args.stride_px
        )

        bev_rgb = draw_bev_overlay(
            rasterize_bev(pts_rgb, cols_rgb, x_range, z_range, args.mpp),
            x_range,
            z_range,
            args.mpp,
            "BEV-RGB",
        )
        bev_seg = draw_bev_overlay(
            rasterize_bev(pts_seg, cols_seg, x_range, z_range, args.mpp),
            x_range,
            z_range,
            args.mpp,
            "BEV-SEG",
        )

        # Also save standalone BEV-seg
        cv2.imwrite(str(outdir / f"{rgb_path.stem}_bev_seg.jpg"), bev_seg)
        cv2.imwrite(str(outdir / f"{rgb_path.stem}_bev_rgb.jpg"), bev_rgb)

        panel = make_panel(bgr, seg_bgr, bev_rgb, bev_seg)
        cv2.imwrite(str(outdir / f"{rgb_path.stem}.jpg"), panel)

        if args.write_video:
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(video_path), fourcc, args.fps, (panel.shape[1], panel.shape[0]))
            writer.write(panel)

        print(
            f"  [{i}/{len(images)}] {rgb_path.name}: masks={n_mask} depth={depth_path.name} "
            f"unit={unit} pts={len(pts_seg)} -> {rgb_path.stem}.jpg"
        )

    if writer is not None:
        writer.release()
        print(f"[bev] video -> {video_path}")

    (outdir / "README.txt").write_text(
        "BEV is a top-down view of the camera/world frame (T_world_cam = I).\n"
        "Axes: OpenCV camera — X right, Y down, Z forward.\n"
        "BEV image: horizontal = +X, vertical up = +Z; yellow cross = camera origin.\n"
        "Segmentation colors are projected with RGB-aligned depth.\n",
        encoding="utf-8",
    )
    print(f"[bev] done -> {outdir}")


if __name__ == "__main__":
    main()
