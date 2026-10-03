"""Milestone 2: build the synthetic multi-celebrity YOLO detection dataset.

Pipeline (used by milestone02_dataset_pipeline.ipynb; also runnable as a script):

1. Source split: every original CelebA photo in data/identities/ is assigned to
   exactly one of train/val/test, reusing the Milestone 1 split. Frames for a
   split are composed only from that split's photos, so no face photo (and no
   frame) is shared between splits.
2. Composition: 2-5 distinct identities per 800x600 frame, arranged as a group
   photo (one or two rows) on a procedurally drawn room background. Photos are
   pasted with a soft elliptical mask, keep their aspect ratio, and vary in
   scale, position and lighting.
3. Annotation: the pretrained YuNet face detector (the method from the team's
   earlier annotation notebook) is run on the finished frame. Every placed
   person must match exactly one detected face (IoU >= 0.5 with the face
   position expected from the source photo), and no unmatched face may remain.
   Frames that fail are regenerated with a new random layout.
4. Augmentation: train frames only, box-aware (albumentations), N copies each.
5. Export: dataset/{train,val,test}/{images,labels}, data.yaml, manifest.csv,
   source_split.csv and summary.json.

Class order is imported from Milestone 1 (project/src/pipeline.py) so the
detector and the classifier use the same class ids.
"""

import argparse
import csv
import hashlib
import json
import shutil
import sys
import urllib.request
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

M2_DIR = Path(__file__).resolve().parent
REPO = M2_DIR.parent
sys.path.insert(0, str(REPO / "project"))
from src.pipeline import IDS  # noqa: E402  (Milestone 1 class order)

DATA_DIR = REPO / "data" / "identities"
M1_SPLIT = REPO / "project" / "configs" / "splits" / "group1_repro_v1_seed47.csv"
OUT_DIR = M2_DIR / "dataset"

SEED = 47
SPLITS = ("train", "val", "test")
BASE_FRAMES = {"train": 168, "val": 36, "test": 36}  # 70 / 15 / 15 of 240 composed frames
AUG_COPIES = 2                                       # augmented copies per train frame
W, H = 800, 600
PEOPLE_CHOICES, PEOPLE_WEIGHTS = (2, 3, 4, 5), (0.25, 0.30, 0.25, 0.20)
JPEG_QUALITY = 90

YUNET_URL = ("https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
YUNET_PATH = M2_DIR / ".cache" / "face_detection_yunet_2023mar.onnx"
SCORE_THRESHOLD, NMS_THRESHOLD = 0.7, 0.3  # a face label needs a detection at >= 0.7
BYSTANDER_SCORE = 0.5                      # any other face-like detection >= 0.5 rejects a frame
MATCH_IOU, EXTRA_FACE_IOU = 0.5, 0.3
MIN_FACE_PX = 36

AUGMENTATIONS = [
    # (transform, parameters, reason) - documented in the notebook and README
    ("HorizontalFlip", "p=0.5", "Faces are left-right symmetric and identity does not depend on side; doubles pose variety."),
    ("Affine", "scale 0.85-1.15, translate +/-5%, rotate +/-7 deg, p=0.8",
     "Camera distance, framing and slight head/camera tilt. Small angles keep boxes tight (ellipse-based box rotation)."),
    ("RandomBrightnessContrast", "brightness +/-0.2, contrast +/-0.2, p=0.7", "Indoor lighting and exposure differences."),
    ("HueSaturationValue", "hue +/-5, saturation +/-15, value +/-10, p=0.4", "White balance and camera colour differences, kept small so skin tones stay plausible."),
    ("RandomGamma", "gamma 85-115, p=0.3", "Non-linear exposure differences."),
    ("GaussianBlur or MotionBlur", "kernel 3-5 / 3-7, p=0.25", "Slight defocus or hand shake in group photos."),
    ("GaussNoise", "std 0.01-0.04, p=0.25", "Sensor noise in dim rooms."),
    ("ImageCompression", "JPEG quality 60-95, p=0.3", "Compression artefacts from phones and messaging apps."),
]
NOT_USED = [
    ("Vertical flip / large rotation", "Upside-down or strongly rotated faces do not occur in group photos and loosen boxes."),
    ("Cutout / random erasing", "Could hide a labelled face and leave a wrong label."),
    ("Mosaic / mixup", "Applied online by YOLOv8 during Milestone 3 training; doing it offline would duplicate it."),
]


# ---------------------------------------------------------------- helpers

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def load_detector():
    if not YUNET_PATH.exists():
        YUNET_PATH.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(YUNET_URL, timeout=60) as response:
            YUNET_PATH.write_bytes(response.read())
    if sha256(YUNET_PATH) != YUNET_SHA256:
        raise ValueError(f"{YUNET_PATH} does not match the pinned YuNet checksum")
    try:  # silence OpenCV's "targets not supported" warning; errors are still shown
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except AttributeError:
        pass
    return cv2.FaceDetectorYN.create(str(YUNET_PATH), "", (320, 320),
                                     BYSTANDER_SCORE, NMS_THRESHOLD, 5000)


def as_saved(rgb):
    """The image exactly as it will be stored (JPEG round trip), so checks see the final pixels."""
    import io
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="JPEG", quality=JPEG_QUALITY)
    return np.asarray(Image.open(buffer).convert("RGB"))


def detect_faces(detector, rgb, min_score=SCORE_THRESHOLD):
    """Return [(x_min, y_min, x_max, y_max, score)] with score >= min_score, in pixels."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    detector.setInputSize((bgr.shape[1], bgr.shape[0]))
    _, faces = detector.detect(bgr)
    if faces is None:
        return []
    return [(float(f[0]), float(f[1]), float(f[0] + f[2]), float(f[1] + f[3]), float(f[14]))
            for f in faces if f[14] >= min_score]


# ---------------------------------------------------------------- 1. source split

def source_split():
    """Assign every photo in data/identities/ to one split, reusing Milestone 1."""
    m1 = {}
    with M1_SPLIT.open(newline="") as handle:
        for row in csv.DictReader(handle):
            m1[row["file"]] = row["split"]
    rows = []
    for identity in IDS:
        files = sorted(p.relative_to(DATA_DIR).as_posix() for p in (DATA_DIR / identity).glob("*.jpg"))
        unused = [f for f in files if f not in m1]
        for f in files:
            if f in m1:
                split, origin = m1[f], "milestone1_split"
            else:  # photos M1 left out when balancing: alternate val/test
                split, origin = ("val", "test")[unused.index(f) % 2], "milestone1_unused"
            rows.append({"identity": identity, "file": f, "split": split, "origin": origin,
                         "sha256": sha256(DATA_DIR / f)})
    return rows


def face_boxes_in_sources(detector, rows):
    """Face box of the main (centred) face in each aligned source photo."""
    boxes = {}
    for row in rows:
        rgb = np.asarray(Image.open(DATA_DIR / row["file"]).convert("RGB"))
        faces = detect_faces(detector, rgb)
        h, w = rgb.shape[:2]
        if not faces:
            raise ValueError(f"No face found in source photo {row['file']}")
        centre = min(faces, key=lambda f: ((f[0] + f[2]) / 2 - w / 2) ** 2 + ((f[1] + f[3]) / 2 - h / 2) ** 2)
        boxes[row["file"]] = centre[:4]
    return boxes


# ---------------------------------------------------------------- 2. composition

WALLS = [(222, 214, 198), (205, 210, 214), (214, 222, 210), (230, 224, 214), (196, 204, 220), (226, 212, 200)]
FLOORS = [(150, 118, 88), (120, 120, 124), (164, 140, 112), (98, 92, 88), (140, 128, 116)]


def make_background(rng):
    """Procedural room: wall gradient, floor, boards/windows/posters, soft light."""
    wall = np.array(WALLS[rng.integers(len(WALLS))], np.float32)
    floor = np.array(FLOORS[rng.integers(len(FLOORS))], np.float32)
    y = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    img = np.broadcast_to(wall * (1.06 - 0.12 * y), (H, W, 3)).copy()
    horizon = int(H * rng.uniform(0.62, 0.78))
    img[horizon:] = floor * (0.9 + 0.2 * (y[horizon:] - y[horizon]) / (1 - y[horizon] + 1e-6))
    img[horizon:horizon + 6] *= 0.75  # skirting board
    for _ in range(rng.integers(1, 4)):
        kind = rng.choice(["board", "window", "poster", "door"])
        bw, bh = {"board": (rng.integers(220, 380), rng.integers(110, 170)),
                  "window": (rng.integers(120, 200), rng.integers(120, 190)),
                  "poster": (rng.integers(60, 110), rng.integers(80, 140)),
                  "door": (rng.integers(90, 120), horizon - 40)}[kind]
        x0 = int(rng.integers(0, W - bw))
        y0 = int(horizon - bh) if kind == "door" else int(rng.integers(30, max(31, horizon - bh - 20)))
        colour = {"board": (238, 240, 236), "window": (176, 204, 228),
                  "poster": tuple(int(c) for c in rng.integers(60, 220, 3)), "door": (128, 96, 70)}[kind]
        img[y0:y0 + bh, x0:x0 + bw] = colour
        frame = 5 if kind in ("board", "window") else 3
        img[y0:y0 + bh, x0:x0 + frame] *= 0.6
        img[y0:y0 + bh, x0 + bw - frame:x0 + bw] *= 0.6
        img[y0:y0 + frame, x0:x0 + bw] *= 0.6
        img[y0 + bh - frame:y0 + bh, x0:x0 + bw] *= 0.6
        if kind == "window":
            img[y0:y0 + bh, x0 + bw // 2 - 2:x0 + bw // 2 + 2] *= 0.7
            img[y0 + bh // 2 - 2:y0 + bh // 2 + 2, x0:x0 + bw] *= 0.7
    # soft light falloff from a random direction
    gx, gy = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H))
    angle = rng.uniform(0, 2 * np.pi)
    img *= (1 + 0.08 * (np.cos(angle) * gx + np.sin(angle) * gy))[..., None]
    img += rng.normal(0, 2.0, img.shape)
    return np.clip(img, 0, 255)


CLOTHES = [(38, 44, 66), (30, 30, 34), (92, 96, 104), (220, 220, 214), (112, 36, 44),
           (190, 172, 140), (52, 82, 60), (66, 96, 140), (150, 90, 60)]


def ellipse_mask(w, h):
    """Soft elliptical alpha around head and hair; hides the rectangular photo border."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((xx - w * 0.5) / (w * 0.43)) ** 2 + ((yy - h * 0.50) / (h * 0.49)) ** 2)
    return np.clip((1.0 - r) / 0.18, 0, 1)


def torso_layer(rng, p):
    """Neck, shoulders and clothing below a pasted head: (rgb, alpha) on the full canvas."""
    cx, top = p["x"] + p["w"] / 2, p["y"] + 0.78 * p["h"]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    depth = np.clip((yy - top) / (0.28 * p["h"]), 0, None)
    half = p["w"] * (0.16 + 0.54 * np.minimum(1.0, depth) ** 0.5)       # neck widening to shoulders
    alpha = np.clip((half - np.abs(xx - cx)) / 4.0, 0, 1) * (yy >= top)
    alpha *= np.clip((yy - top) / 6.0, 0, 1)
    colour = np.array(CLOTHES[rng.integers(len(CLOTHES))], np.float32) * rng.uniform(0.85, 1.1)
    shade = (1.0 - 0.18 * np.clip((yy - top) / p["h"], 0, 1))[..., None]
    side = (1.0 - 0.12 * np.abs(xx - cx) / (0.7 * p["w"]))[..., None]
    return np.clip(colour * shade * side, 0, 255), alpha


def draw_table(rng, canvas, top):
    """A desk edge in front of the front row (classroom look)."""
    wood = np.array(FLOORS[rng.integers(len(FLOORS))], np.float32) * rng.uniform(1.0, 1.25)
    y = np.arange(top, H, dtype=np.float32)[:, None, None]
    canvas[top:] = np.clip(wood * (1.0 - 0.15 * (y - top) / max(1, H - top)), 0, 255)
    canvas[top:top + 5] *= 1.15
    canvas[top + 5:top + 10] *= 0.7


def _spot(row, cx, bottom, h):
    w = h * 178 / 218  # keep the CelebA aspect ratio
    return {"row": row, "x": int(round(cx - w / 2)), "y": int(round(bottom - h)),
            "w": int(round(w)), "h": int(round(h))}


def plan_layout(rng, n):
    """Positions for a one-row or two-row group photo (back row smaller, in the gaps)."""
    two_rows = n >= 4 and rng.random() < 0.8
    n_front = n - n // 2 if two_rows else n
    n_back = n - n_front
    distance = rng.uniform(0.62, 1.0) if (two_rows or n <= 3) else rng.uniform(0.62, 0.8)
    front_h = rng.uniform(250, 330) * distance * (rng.uniform(1.0, 1.15) if n_front <= 2 else 1.0)
    front_bottom = H + rng.uniform(-10, 60) if distance > 0.85 else rng.uniform(0.8, 1.0) * H
    spacing = W / n_front
    front_cx = [(i + 0.5) * spacing + rng.uniform(-0.12, 0.12) * spacing for i in range(n_front)]
    people = [_spot("front", cx, front_bottom - rng.uniform(0, 30), front_h * rng.uniform(0.92, 1.08))
              for cx in front_cx]
    if n_back:
        back_h = front_h * rng.uniform(0.78, 0.9)
        margin = 0.6 * back_h * 178 / 218 / 2
        gaps = [(a + b) / 2 for a, b in zip(front_cx, front_cx[1:])]
        gaps += [front_cx[0] - spacing / 2, front_cx[-1] + spacing / 2]
        gaps = [min(max(g, margin), W - margin) for g in gaps]
        back_bottom = front_bottom - front_h * rng.uniform(0.55, 0.7)
        for j in sorted(rng.choice(len(gaps), size=n_back, replace=False)):
            people.append(_spot("back", gaps[j] + rng.uniform(-0.08, 0.08) * spacing,
                                back_bottom - rng.uniform(0, 20), back_h * rng.uniform(0.95, 1.05)))
    return people


def compose_frame(rng, pool, source_faces, n):
    """Compose one frame; returns (rgb uint8, people) or None if the layout is unusable."""
    identities = [str(i) for i in rng.choice(IDS, size=n, replace=False)]
    layout = plan_layout(rng, n)
    paint_order = sorted(range(n), key=lambda k: (layout[k]["row"] == "front", layout[k]["y"] + layout[k]["h"]))
    canvas = make_background(rng)
    alphas = []
    people = []
    for k in range(n):
        spot = layout[k]
        source = pool[identities[k]][rng.integers(len(pool[identities[k]]))]
        s = spot["h"] / 218
        fx0, fy0, fx1, fy1 = source_faces[source]
        expected = (spot["x"] + fx0 * s, spot["y"] + fy0 * s, spot["x"] + fx1 * s, spot["y"] + fy1 * s)
        if (expected[0] < 2 or expected[1] < 2 or expected[2] > W - 2 or expected[3] > H - 2
                or expected[2] - expected[0] < MIN_FACE_PX):
            return None  # face would be cut by the frame or too small
        people.append({**spot, "identity": identities[k], "class_id": IDS.index(identities[k]),
                       "source_file": source, "expected_box": expected})
    # paint back to front; check later-painted people never cover an earlier face
    full_alpha = []
    for k in paint_order:
        p = people[k]
        body, body_alpha = torso_layer(rng, p)
        canvas = body * body_alpha[..., None] + canvas * (1 - body_alpha[..., None])
        photo = Image.open(DATA_DIR / p["source_file"]).convert("RGB").resize((p["w"], p["h"]), Image.LANCZOS)
        photo = np.asarray(photo, np.float32)
        tint = 1 + rng.uniform(-0.04, 0.04, 3)
        photo = np.clip(photo * rng.uniform(0.85, 1.15) * tint, 0, 255)
        alpha = ellipse_mask(p["w"], p["h"])
        layer = np.zeros((H, W), np.float32)
        x0, y0 = max(p["x"], 0), max(p["y"], 0)
        x1, y1 = min(p["x"] + p["w"], W), min(p["y"] + p["h"], H)
        if x1 <= x0 or y1 <= y0:
            return None
        crop_a = alpha[y0 - p["y"]:y1 - p["y"], x0 - p["x"]:x1 - p["x"]]
        crop_p = photo[y0 - p["y"]:y1 - p["y"], x0 - p["x"]:x1 - p["x"]]
        layer[y0:y1, x0:x1] = crop_a
        layer = np.maximum(layer, body_alpha)
        canvas[y0:y1, x0:x1] = crop_p * crop_a[..., None] + canvas[y0:y1, x0:x1] * (1 - crop_a[..., None])
        full_alpha.append((k, layer))
    for idx, (k, _) in enumerate(full_alpha):
        ex0, ey0, ex1, ey1 = (int(v) for v in people[k]["expected_box"])
        for _, later in full_alpha[idx + 1:]:
            if (later[ey0:ey1, ex0:ex1] > 0.3).mean() > 0.03:
                return None  # a face would be occluded
    if rng.random() < 0.4:  # desk in front of the front row, below every face
        lowest_face = max(p["expected_box"][3] for p in people)
        top = int(lowest_face + rng.uniform(0.35, 0.6) * max(p["h"] for p in people if p["row"] == "front"))
        if top < H - 20:
            draw_table(rng, canvas, top)
    # global lighting of the "photograph"
    canvas = canvas * rng.uniform(0.9, 1.1) + rng.uniform(-10, 10)
    canvas = canvas * (1 + np.array([rng.uniform(-0.04, 0.04), 0, rng.uniform(-0.04, 0.04)]))
    rgb = np.clip(canvas, 0, 255).astype(np.uint8)
    sigma = rng.uniform(0, 0.8)
    if sigma > 0.2:
        rgb = cv2.GaussianBlur(rgb, (0, 0), sigma)
    return rgb, people


# ---------------------------------------------------------------- 3. annotation

def annotate(detector, rgb, people):
    """Match YuNet detections to placed people; None if any face is missing or extra."""
    saved = as_saved(rgb)
    detections = detect_faces(detector, saved)                    # >= 0.7: candidate labels
    candidates = detect_faces(detector, saved, BYSTANDER_SCORE)    # >= 0.5: anything face-like
    used = set()
    for p in people:
        scored = [(iou(p["expected_box"], d[:4]), j) for j, d in enumerate(detections) if j not in used]
        best = max(scored, default=(0, None))
        if best[0] < MATCH_IOU:
            return None
        used.add(best[1])
        x0, y0, x1, y1, score = detections[best[1]]
        x0, y0, x1, y1 = max(0.0, x0), max(0.0, y0), min(float(W), x1), min(float(H), y1)
        p.update(box=(x0, y0, x1, y1), confidence=score, match_iou=best[0])
    for d in candidates:
        if max(iou(d[:4], p["box"]) for p in people) < EXTRA_FACE_IOU:
            return None  # an unlabelled (bystander) face is visible
    return people


def yolo_rows(people, width=W, height=H):
    return [(p["class_id"], (p["box"][0] + p["box"][2]) / 2 / width, (p["box"][1] + p["box"][3]) / 2 / height,
             (p["box"][2] - p["box"][0]) / width, (p["box"][3] - p["box"][1]) / height) for p in people]


def build_frames(detector, split, pool, source_faces, count, log=print):
    frames, attempts = [], Counter()
    for index in range(count):
        for attempt in range(200):
            rng = np.random.default_rng([SEED, SPLITS.index(split), index, attempt])
            n = int(rng.choice(PEOPLE_CHOICES, p=PEOPLE_WEIGHTS))
            composed = compose_frame(rng, pool, source_faces, n)
            if composed is None:
                attempts["layout_rejected"] += 1
                continue
            rgb, people = composed
            if annotate(detector, rgb, people) is None:
                attempts["annotation_rejected"] += 1
                continue
            frames.append({"name": f"{split}_{index:04d}", "rgb": rgb, "people": people, "seed": [SEED, SPLITS.index(split), index, attempt]})
            break
        else:
            raise RuntimeError(f"Could not compose {split} frame {index}")
    log(f"{split}: {count} frames ({attempts['layout_rejected']} layouts and "
        f"{attempts['annotation_rejected']} annotations rejected and regenerated)")
    return frames, attempts


# ---------------------------------------------------------------- 4. augmentation

def augmentation_steps():
    """The transforms in AUGMENTATIONS, in order."""
    import albumentations as A
    return [
        A.HorizontalFlip(p=0.5),
        A.Affine(scale=(0.85, 1.15), translate_percent=(-0.05, 0.05), rotate=(-7, 7),
                 rotate_method="ellipse", border_mode=cv2.BORDER_CONSTANT, fill=(114, 114, 114), p=0.8),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=15, val_shift_limit=10, p=0.4),
        A.RandomGamma(gamma_limit=(85, 115), p=0.3),
        A.OneOf([A.GaussianBlur(blur_limit=(3, 5)), A.MotionBlur(blur_limit=(3, 7))], p=0.25),
        A.GaussNoise(std_range=(0.01, 0.04), p=0.25),
        A.ImageCompression(quality_range=(60, 95), p=0.3),
    ]


def augment_once(image, boxes, labels, seed):
    """Apply each transform with its own seed.

    A single seeded A.Compose (albumentations 2.0.8) draws the same random number
    for every transform, so the transforms were applied all together or not at
    all. Seeding each step separately keeps them independent and reproducible.
    """
    import albumentations as A
    for step, transform in enumerate(augmentation_steps()):
        if isinstance(transform, A.DualTransform):  # geometric: move the boxes too
            out = A.Compose([transform], seed=seed * 16 + step, bbox_params=A.BboxParams(
                format="yolo", label_fields=["class_labels"], min_visibility=0.95, clip=True))(
                image=image, bboxes=boxes, class_labels=labels)
            image, boxes, labels = out["image"], out["bboxes"], out["class_labels"]
        else:  # pixel-level: boxes unchanged
            image = A.Compose([transform], seed=seed * 16 + step)(image=image)["image"]
    return image, boxes, labels


def augment_frame(frame, copy, existing=(), detector=None):
    rows = yolo_rows(frame["people"])
    boxes = [r[1:] for r in rows]
    labels = [r[0] for r in rows]
    for attempt in range(20):
        seed = SEED * 100_000 + int(frame["name"].split("_")[1]) * 100 + copy * 20 + attempt
        image, out_boxes, out_labels = augment_once(frame["rgb"], boxes, labels, seed)
        new = [(int(c), *map(float, b)) for c, b in zip(out_labels, out_boxes)]
        unchanged = any(np.abs(image.astype(np.int16) - e).mean() < 2.0 for e in (frame["rgb"], *existing))
        if (not unchanged and sorted(r[0] for r in new) == sorted(labels)
                and all(r[3] * W >= MIN_FACE_PX * 0.8 for r in new)
                and (detector is None or no_unlabelled_face(detector, image, new))):
            return image, new, attempt
    raise RuntimeError(f"No valid augmentation found for {frame['name']}")


def draw_boxes(rgb, rows, ax=None, title=None):
    """Draw YOLO rows (class, xc, yc, w, h) on an image with matplotlib."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    ax = ax or plt.gca()
    ax.imshow(rgb)
    height, width = rgb.shape[:2]
    colours = plt.get_cmap("tab10")
    for c, x, y, w, h in rows:
        x0, y0 = (x - w / 2) * width, (y - h / 2) * height
        ax.add_patch(Rectangle((x0, y0), w * width, h * height, fill=False, edgecolor=colours(c), linewidth=2))
        ax.text(x0, y0 - 3, f"{c}:{IDS[c]}", color="white", fontsize=7,
                bbox={"facecolor": colours(c), "alpha": 0.9, "pad": 1, "linewidth": 0})
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=9)


def read_label(path):
    return [(int(r[0]), *map(float, r[1:])) for r in (line.split() for line in Path(path).read_text().splitlines())]


def no_unlabelled_face(detector, rgb, rows):
    """True if every face-like detection (>= 0.5) in the saved image overlaps a label."""
    rgb = as_saved(rgb)
    height, width = rgb.shape[:2]
    boxes = [((x - w / 2) * width, (y - h / 2) * height, (x + w / 2) * width, (y + h / 2) * height)
             for _, x, y, w, h in rows]
    return all(max(iou(d[:4], b) for b in boxes) >= EXTRA_FACE_IOU
               for d in detect_faces(detector, rgb, BYSTANDER_SCORE))


# ---------------------------------------------------------------- 5. export

def write_label(path, rows):
    path.write_text("".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in rows))


def export(frames_by_split, sources, detector=None, out_dir=OUT_DIR, log=print):
    if out_dir.exists():
        shutil.rmtree(out_dir)
    manifest, aug_retries = [], 0
    for split in SPLITS:
        (out_dir / split / "images").mkdir(parents=True)
        (out_dir / split / "labels").mkdir(parents=True)
        for frame in frames_by_split[split]:
            variants = [(frame["name"], frame["rgb"], yolo_rows(frame["people"]), "")]
            if split == "train":
                for copy in range(1, AUG_COPIES + 1):
                    image, rows, retries = augment_frame(frame, copy, [v[1] for v in variants[1:]], detector)
                    aug_retries += retries
                    variants.append((f"{frame['name']}_aug{copy}", image, rows, frame["name"]))
            for name, image, rows, parent in variants:
                Image.fromarray(image).save(out_dir / split / "images" / f"{name}.jpg", quality=JPEG_QUALITY)
                write_label(out_dir / split / "labels" / f"{name}.txt", rows)
                by_class = {q["class_id"]: q for q in frame["people"]}  # identities are distinct per frame
                for c, x, y, w, h in rows:
                    p = by_class[c]
                    manifest.append({"split": split, "image": f"{split}/images/{name}.jpg", "augmented_from": parent,
                                     "identity": IDS[c], "class_id": c, "source_file": p["source_file"],
                                     "row": p["row"], "x_center": round(x, 6), "y_center": round(y, 6),
                                     "width": round(w, 6), "height": round(h, 6),
                                     "detector_confidence": round(p["confidence"], 4) if not parent else "",
                                     "frame_seed": "-".join(map(str, frame["seed"]))})
    with (out_dir / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(manifest)
    with (out_dir / "source_split.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sources[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(sources)
    names = "\n".join(f"  {i}: '{identity}'" for i, identity in enumerate(IDS))
    (out_dir / "data.yaml").write_text(
        "# YOLOv8 dataset config. Paths are relative to this file's folder.\n"
        "train: train/images\nval: val/images\ntest: test/images\n"
        f"nc: {len(IDS)}\nnames:\n{names}\n")
    log(f"augmentation resampled {aug_retries} times (a face left the frame, a bystander face appeared, or the copy was unchanged)")
    return manifest


def validate(out_dir=OUT_DIR):
    """Independent checks on the exported files; returns a summary dict."""
    import pandas as pd
    m = pd.read_csv(out_dir / "manifest.csv", dtype={"identity": str, "source_file": str})
    src = pd.read_csv(out_dir / "source_split.csv", dtype=str)
    summary = {"splits": {}}
    hashes = {}
    for split in SPLITS:
        images = sorted((out_dir / split / "images").glob("*.jpg"))
        labels = sorted((out_dir / split / "labels").glob("*.txt"))
        assert [p.stem for p in images] == [p.stem for p in labels], f"{split}: image/label mismatch"
        faces = Counter()
        for img, lab in zip(images, labels):
            with Image.open(img) as im:
                assert im.size == (W, H)
            rows = [line.split() for line in lab.read_text().splitlines()]
            assert rows, f"{lab} is empty"
            for r in rows:
                assert len(r) == 5
                c, x, y, w, h = int(r[0]), *map(float, r[1:])
                assert 0 <= c < len(IDS)
                assert 0 <= x - w / 2 + 1e-6 and x + w / 2 <= 1 + 1e-6 and 0 <= y - h / 2 + 1e-6 and y + h / 2 <= 1 + 1e-6
                faces[IDS[c]] += 1
            digest = sha256(img)
            assert digest not in hashes, f"duplicate image {img} and {hashes[digest]}"
            hashes[digest] = img
        base = sum(1 for p in images if "_aug" not in p.stem)
        summary["splits"][split] = {"images": len(images), "composed": base, "augmented": len(images) - base,
                                    "faces": sum(faces.values()), "faces_per_identity": {i: faces[i] for i in IDS}}
        assert all(faces[i] > 0 for i in IDS), f"{split}: an identity is missing"
    used = m.groupby("split").source_file.apply(set)
    for a in SPLITS:
        for b in SPLITS:
            if a < b:
                assert not (used.get(a, set()) & used.get(b, set())), f"source photo shared by {a} and {b}"
    for split in SPLITS:
        allowed = set(src[src.split == split].file)
        assert used.get(split, set()) <= allowed, f"{split} uses a photo from another split"
    summary["source_photos_used"] = {s: len(used.get(s, set())) for s in SPLITS}
    summary["source_photos_shared_between_splits"] = 0
    summary["duplicate_images"] = 0
    composed = sum(v["composed"] for v in summary["splits"].values())
    summary["composed_proportions"] = {s: round(summary["splits"][s]["composed"] / composed, 3) for s in SPLITS}
    summary["dataset_mb"] = round(sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file()) / 1e6, 1)
    return summary


def build(log=print):
    detector = load_detector()
    sources = source_split()
    source_faces = face_boxes_in_sources(detector, sources)
    frames, attempts = {}, {}
    for split in SPLITS:
        pool = {i: [r["file"] for r in sources if r["split"] == split and r["identity"] == i] for i in IDS}
        frames[split], attempts[split] = build_frames(detector, split, pool, source_faces, BASE_FRAMES[split], log)
    export(frames, sources, detector, log=log)
    summary = validate()
    summary["regenerated"] = {s: dict(a) for s, a in attempts.items()}
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return frames, sources, summary


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    print(json.dumps(build()[2], indent=2))
