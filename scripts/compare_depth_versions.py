#!/usr/bin/env python3
"""Compare Depth Anything V1 / V2 / V3 on the same RGB frames (viz + vs GT metrics)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from depth_backend import (  # noqa: E402
    DEFAULT_V2_HF,
    DEFAULT_V3_HF,
    LEGACY_V1_HF,
    load_depth_backend,
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="Depth Anything V1/V2/V3 comparison")
    p.add_argument("--source", type=Path, default=root / "data" / "jj_salon_room" / "rgb")
    p.add_argument("--gt-depth-dir", type=Path, default=root / "data" / "jj_salon_room" / "depth")
    p.add_argument("--outdir", type=Path, default=root / "tmp" / "depth_version_compare")
    p.add_argument("--num-frames", type=int, default=8)
    p.add_argument("--stride", type=int, default=50)
    p.add_argument("--device", default="0")
    p.add_argument("--v1-weights", type=str, default=LEGACY_V1_HF)
    p.add_argument("--v2-weights", type=str, default=DEFAULT_V2_HF)
    p.add_argument("--v3-weights", type=str, default=DEFAULT_V3_HF)
    p.add_argument(
        "--skip",
        type=str,
        default="",
        help="Comma versions to skip, e.g. v3",
    )
    p.add_argument("--write-video", action="store_true")
    p.add_argument("--fps", type=float, default=4.0)
    return p.parse_args()


def collect_images(source: Path, num_frames: int, stride: int) -> list[Path]:
    files = sorted(p for p in source.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    files = files[:: max(stride, 1)]
    if num_frames > 0:
        files = files[:num_frames]
    if not files:
        raise FileNotFoundError(f"no images under {source}")
    return files


def colorize(depth: np.ndarray) -> np.ndarray:
    d = depth.astype(np.float32)
    finite = np.isfinite(d)
    if not finite.any():
        return np.zeros((*d.shape, 3), dtype=np.uint8)
    lo, hi = np.nanpercentile(d[finite], [1, 99])
    if hi <= lo:
        hi = lo + 1e-6
    u8 = (np.clip((d - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    u8[~finite] = 0
    return cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)


def load_gt_m(path: Path, wh: tuple[int, int]) -> np.ndarray:
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        return np.zeros((wh[1], wh[0]), dtype=np.float32)
    d = raw.astype(np.float32) / 1000.0
    w, h = wh
    if d.shape[0] != h or d.shape[1] != w:
        d = cv2.resize(d, (w, h), interpolation=cv2.INTER_NEAREST)
    return d


def metricize(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, dict]:
    """Align relative pred to GT metres; auto detect disparity vs depth-like."""
    valid = (gt > 0.2) & (gt < 5.0) & np.isfinite(pred) & (pred > 1e-4)
    info = {"n_valid": int(valid.sum()), "mode": "none", "corr": None, "scale": None}
    if valid.sum() < 100:
        return pred.copy(), info
    corr = float(np.corrcoef(pred[valid], gt[valid])[0, 1])
    info["corr"] = corr
    if corr < 0.0:
        scale = float(np.median(gt[valid] * pred[valid]))
        out = scale / np.clip(pred, 1e-6, None)
        info["mode"] = "disp"
        info["scale"] = scale
    else:
        scale = float(np.median(gt[valid] / pred[valid]))
        out = pred * scale
        info["mode"] = "depth"
        info["scale"] = scale
    return out, info


def depth_metrics(pred_m: np.ndarray, gt: np.ndarray) -> dict:
    valid = (gt > 0.2) & (gt < 5.0) & np.isfinite(pred_m) & (pred_m > 0.05) & (pred_m < 8.0)
    if valid.sum() < 100:
        return {"n": int(valid.sum())}
    err = pred_m[valid] - gt[valid]
    abs_rel = np.mean(np.abs(err) / gt[valid])
    rmse = float(np.sqrt(np.mean(err**2)))
    # delta1
    ratio = np.maximum(pred_m[valid] / gt[valid], gt[valid] / pred_m[valid])
    d1 = float(np.mean(ratio < 1.25))
    return {
        "n": int(valid.sum()),
        "abs_rel": float(abs_rel),
        "rmse_m": rmse,
        "mae_m": float(np.mean(np.abs(err))),
        "delta1": d1,
        "median_err_m": float(np.median(err)),
    }


def label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, text, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def make_row(images: list[np.ndarray], titles: list[str]) -> np.ndarray:
    h = min(im.shape[0] for im in images)
    resized = []
    for im, t in zip(images, titles):
        scale = h / im.shape[0]
        r = cv2.resize(im, (int(im.shape[1] * scale), h))
        resized.append(label(r, t))
    gap = np.full((h, 8, 3), 30, dtype=np.uint8)
    out = resized[0]
    for r in resized[1:]:
        out = np.concatenate([out, gap, r], axis=1)
    return out


def main() -> None:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    args = parse_args()
    skip = {s.strip().lower() for s in args.skip.split(",") if s.strip()}

    versions = [
        ("v1", args.v1_weights),
        ("v2", args.v2_weights),
        ("v3", args.v3_weights),
    ]
    versions = [(v, w) for v, w in versions if v not in skip]

    images = collect_images(args.source, args.num_frames, args.stride)
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    raw_dir = outdir / "raw"
    raw_dir.mkdir(exist_ok=True)

    backends = {}
    for ver, weights in versions:
        print(f"[compare] loading {ver}: {weights}", flush=True)
        backends[ver] = load_depth_backend(ver, weights, args.device)
        print(f"[compare] ready {ver} @ {backends[ver].device}", flush=True)

    all_frame_metrics: dict[str, list[dict]] = {v: [] for v, _ in versions}
    writer = None
    video_path = outdir / "depth_v1_v2_v3_compare.mp4"

    for i, path in enumerate(images, 1):
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"failed to read {path}")
        h, w = bgr.shape[:2]
        gt_path = args.gt_depth_dir / f"{path.stem}.png"
        gt = load_gt_m(gt_path, (w, h)) if gt_path.is_file() else np.zeros((h, w), np.float32)

        preds_raw = {}
        preds_m = {}
        panels_abs = [bgr]
        titles_abs = ["RGB"]
        panels_err = []
        titles_err = []

        for ver, _ in versions:
            pred = backends[ver].predict(bgr)
            preds_raw[ver] = pred
            np.save(raw_dir / f"{path.stem}_{ver}.npy", pred.astype(np.float16))
            pred_m, align = metricize(pred, gt)
            preds_m[ver] = pred_m
            met = depth_metrics(pred_m, gt)
            met.update({"align_mode": align["mode"], "corr_raw": align["corr"]})
            all_frame_metrics[ver].append(met)

            tag = f"{ver.upper()}"
            if met.get("abs_rel") is not None:
                tag += f" absRel={met['abs_rel']:.3f} d1={met['delta1']:.2f}"
            panels_abs.append(colorize(pred))
            titles_abs.append(tag)

            if gt.any():
                err = np.abs(pred_m - gt)
                err_vis = err.copy()
                err_vis[~((gt > 0.2) & (gt < 5))] = np.nan
                panels_err.append(colorize(np.nan_to_num(err_vis, nan=0.0)))
                titles_err.append(f"|err| {ver.upper()}")

        # Absolute/relative colored row
        row1 = make_row(panels_abs, titles_abs)
        if panels_err:
            # prepend GT colorized
            row2 = make_row([colorize(gt)] + panels_err, ["GT depth"] + titles_err)
            # match widths
            if row1.shape[1] != row2.shape[1]:
                tw = max(row1.shape[1], row2.shape[1])
                def pad(im, ww):
                    if im.shape[1] == ww:
                        return im
                    return cv2.copyMakeBorder(im, 0, 0, 0, ww - im.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0))
                row1, row2 = pad(row1, tw), pad(row2, tw)
            panel = np.vstack([row1, row2])
        else:
            panel = row1

        # shrink if huge
        if panel.shape[1] > 3840:
            s = 3840 / panel.shape[1]
            panel = cv2.resize(panel, (3840, int(panel.shape[0] * s)))

        out_jpg = outdir / f"{path.stem}.jpg"
        cv2.imwrite(str(out_jpg), panel)

        if args.write_video:
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(video_path), fourcc, args.fps, (panel.shape[1], panel.shape[0]))
            writer.write(panel)

        msg = f"  [{i}/{len(images)}] {path.name}"
        for ver, _ in versions:
            m = all_frame_metrics[ver][-1]
            if "abs_rel" in m:
                msg += f" | {ver}: absRel={m['abs_rel']:.3f} rmse={m['rmse_m']:.3f} d1={m['delta1']:.3f} ({m['align_mode']})"
            else:
                msg += f" | {ver}: n={m.get('n', 0)}"
        print(msg, flush=True)

    if writer is not None:
        writer.release()
        print(f"[compare] video -> {video_path}")

    # Aggregate report
    report = {"frames": [p.name for p in images], "versions": {}}
    lines = ["# Depth Anything V1 / V2 / V3 comparison", ""]
    lines.append(f"- frames: {len(images)} (stride={args.stride})")
    lines.append(f"- metricize: auto (corr<0 → scale/pred else pred*scale) vs GT depth mm")
    lines.append("")
    lines.append("| version | abs_rel↓ | rmse_m↓ | mae_m↓ | delta1↑ | align |")
    lines.append("|---|---:|---:|---:|---:|---|")

    for ver, weights in versions:
        mets = [m for m in all_frame_metrics[ver] if "abs_rel" in m]
        if not mets:
            report["versions"][ver] = {"weights": weights, "n_frames_with_gt": 0}
            lines.append(f"| {ver} | - | - | - | - | no GT overlap |")
            continue
        summary = {
            "weights": weights,
            "n_frames_with_gt": len(mets),
            "abs_rel_mean": float(np.mean([m["abs_rel"] for m in mets])),
            "rmse_m_mean": float(np.mean([m["rmse_m"] for m in mets])),
            "mae_m_mean": float(np.mean([m["mae_m"] for m in mets])),
            "delta1_mean": float(np.mean([m["delta1"] for m in mets])),
            "align_modes": sorted({m["align_mode"] for m in mets}),
            "corr_raw_mean": float(np.mean([m["corr_raw"] for m in mets if m["corr_raw"] is not None])),
            "per_frame": mets,
        }
        report["versions"][ver] = summary
        lines.append(
            f"| **{ver}** | {summary['abs_rel_mean']:.4f} | {summary['rmse_m_mean']:.3f} | "
            f"{summary['mae_m_mean']:.3f} | {summary['delta1_mean']:.3f} | "
            f"{','.join(summary['align_modes'])} |"
        )

    lines.append("")
    lines.append("Lower abs_rel / rmse / mae is better; higher delta1 is better.")
    lines.append("Visual panels: top = relative depth colorized; bottom = |metricized-GT| error (where GT valid).")
    lines.append(f"Outputs: `{outdir}`")

    (outdir / "compare_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (outdir / "compare_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"[compare] done -> {outdir}")


if __name__ == "__main__":
    main()
