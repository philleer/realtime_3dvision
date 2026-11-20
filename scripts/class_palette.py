#!/usr/bin/env python3
"""Fixed, high-contrast class colors shared by SEG overlays and BEV.

COCO-80 ids → BGR uint8. Nearby ids and common indoor classes are spaced in hue.
"""

from __future__ import annotations

import cv2
import numpy as np

# Hand-tuned BGR for COCO-80. Prefer saturated, mutually distant colors.
# Common indoor: person=0, chair=56, couch=57, potted plant=58, bed=59,
# dining table=60, toilet=61, tv=62, laptop=63, book=73, clock=74, vase=75.
# BGR tuples. Indoor-critical ids intentionally far apart in hue:
# person=crimson, chair=green, couch=orange, plant=blue, bed=magenta,
# table=teal, tv=violet, book=gold, clock=cyan.
_COCO80_BGR: list[tuple[int, int, int]] = [
    (60, 20, 220),  # 0 person — crimson red
    (0, 200, 255),  # 1 bicycle — amber
    (255, 80, 0),  # 2 car — blue
    (0, 255, 128),  # 3 motorcycle — spring green
    (200, 0, 200),  # 4 airplane — magenta
    (0, 165, 255),  # 5 bus — orange
    (180, 255, 0),  # 6 train — cyan-green
    (255, 0, 128),  # 7 truck — purple-blue
    (50, 255, 50),  # 8 boat — lime
    (128, 0, 255),  # 9 traffic light — violet-red
    (255, 200, 0),  # 10 fire hydrant
    (0, 100, 255),  # 11 stop sign
    (100, 255, 200),  # 12 parking meter
    (255, 50, 150),  # 13 bench
    (80, 180, 255),  # 14 bird
    (200, 255, 100),  # 15 cat
    (255, 100, 50),  # 16 dog
    (150, 50, 255),  # 17 horse
    (50, 255, 200),  # 18 sheep
    (255, 150, 0),  # 19 cow
    (100, 100, 255),  # 20 elephant
    (0, 220, 180),  # 21 bear
    (220, 0, 100),  # 22 zebra
    (0, 255, 255),  # 23 giraffe — yellow
    (180, 100, 0),  # 24 backpack
    (0, 80, 200),  # 25 umbrella
    (200, 200, 0),  # 26 handbag
    (100, 0, 180),  # 27 tie
    (0, 200, 100),  # 28 suitcase
    (255, 0, 80),  # 29 frisbee
    (80, 255, 0),  # 30 skis
    (200, 50, 50),  # 31 snowboard
    (50, 50, 255),  # 32 sports ball
    (255, 255, 100),  # 33 kite
    (100, 200, 0),  # 34 baseball bat
    (0, 150, 255),  # 35 baseball glove
    (255, 0, 200),  # 36 skateboard
    (0, 255, 50),  # 37 surfboard
    (150, 0, 100),  # 38 tennis racket
    (50, 200, 255),  # 39 bottle
    (200, 100, 255),  # 40 wine glass
    (100, 255, 50),  # 41 cup
    (255, 80, 80),  # 42 fork
    (80, 80, 200),  # 43 knife
    (200, 255, 50),  # 44 spoon
    (50, 150, 50),  # 45 bowl
    (255, 50, 255),  # 46 banana
    (50, 255, 150),  # 47 apple
    (150, 50, 50),  # 48 sandwich
    (50, 50, 150),  # 49 orange
    (255, 200, 100),  # 50 broccoli
    (100, 50, 200),  # 51 carrot
    (200, 255, 200),  # 52 hot dog
    (200, 50, 150),  # 53 pizza
    (50, 200, 50),  # 54 donut
    (150, 150, 255),  # 55 cake
    (0, 210, 40),  # 56 chair — vivid green
    (0, 140, 255),  # 57 couch — vivid orange
    (255, 90, 40),  # 58 potted plant — strong blue
    (200, 40, 180),  # 59 bed — magenta/purple
    (180, 200, 0),  # 60 dining table — teal/cyan-green
    (220, 220, 40),  # 61 toilet
    (255, 0, 220),  # 62 tv — violet/magenta
    (40, 240, 240),  # 63 laptop — light yellow
    (255, 120, 0),  # 64 mouse
    (160, 40, 255),  # 65 remote — purple-pink
    (0, 255, 180),  # 66 keyboard
    (40, 180, 220),  # 67 cell phone — coral
    (50, 100, 255),  # 68 microwave
    (255, 50, 0),  # 69 oven
    (0, 120, 120),  # 70 toaster
    (120, 255, 120),  # 71 sink
    (120, 120, 0),  # 72 refrigerator
    (0, 215, 255),  # 73 book — gold/yellow (≠ person red)
    (255, 255, 0),  # 74 clock — cyan
    (180, 0, 255),  # 75 vase — pink-purple
    (0, 180, 255),  # 76 scissors
    (255, 180, 180),  # 77 teddy bear
    (180, 255, 255),  # 78 hair drier
    (100, 100, 0),  # 79 toothbrush
]

assert len(_COCO80_BGR) == 80

# Fallback for non-COCO / out-of-range ids (deterministic, spaced)
_FALLBACK = np.array(
    [
        [255, 64, 64],
        [64, 255, 64],
        [64, 64, 255],
        [255, 255, 64],
        [255, 64, 255],
        [64, 255, 255],
        [200, 120, 40],
        [40, 200, 120],
        [120, 40, 200],
        [200, 40, 120],
    ],
    dtype=np.uint8,
)


def class_bgr(cls_id: int) -> tuple[int, int, int]:
    cid = int(cls_id)
    if 0 <= cid < 80:
        return _COCO80_BGR[cid]
    c = _FALLBACK[cid % len(_FALLBACK)]
    return int(c[0]), int(c[1]), int(c[2])


def class_bgr_arr(cls_id: int) -> np.ndarray:
    return np.array(class_bgr(cls_id), dtype=np.uint8)


def instances_from_yolo(result, h: int, w: int) -> list[dict]:
    """Build [{cls, conf, name, mask}] from an Ultralytics seg result."""
    if result.masks is None or len(result.masks) == 0:
        return []
    masks = result.masks.data.detach().cpu().numpy()
    names = result.names if hasattr(result, "names") else {}
    out: list[dict] = []
    for i in range(masks.shape[0]):
        m = masks[i]
        if m.shape[0] != h or m.shape[1] != w:
            m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        bin_m = m > 0.5
        if not bin_m.any():
            continue
        cls_id = i
        conf = 1.0
        if result.boxes is not None and i < len(result.boxes):
            if result.boxes.cls is not None:
                cls_id = int(result.boxes.cls[i].item())
            if result.boxes.conf is not None:
                conf = float(result.boxes.conf[i].item())
        out.append(
            {
                "cls": cls_id,
                "conf": conf,
                "name": str(names.get(cls_id, cls_id)),
                "mask": bin_m,
            }
        )
    return out


def paint_instances(
    bgr: np.ndarray,
    instances: list[dict],
    *,
    alpha: float = 0.55,
    draw_boxes: bool = True,
    draw_labels: bool = True,
) -> np.ndarray:
    """Paint masks/boxes/labels with the shared palette. instances: cls, conf, name, mask."""
    out = bgr.copy()
    h, w = out.shape[:2]
    # Higher conf last so it stays visible
    ordered = sorted(instances, key=lambda d: float(d.get("conf", 0.0)))
    for inst in ordered:
        mask = inst["mask"]
        if mask.shape[:2] != (h, w):
            continue
        color = class_bgr(inst["cls"])
        out[mask] = (
            (1.0 - alpha) * out[mask].astype(np.float32) + alpha * np.array(color, np.float32)
        ).astype(np.uint8)
        if draw_boxes:
            ys, xs = np.where(mask)
            if len(xs) == 0:
                continue
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            cv2.rectangle(out, (x0, y0), (x1, y1), color, 2)
            if draw_labels:
                label = f"{inst.get('name', inst['cls'])} {float(inst.get('conf', 0)):.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                y_text = max(0, y0 - 6)
                cv2.rectangle(out, (x0, y_text - th - 4), (x0 + tw + 4, y_text + 2), color, -1)
                cv2.putText(
                    out,
                    label,
                    (x0 + 2, y_text - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
    return out
