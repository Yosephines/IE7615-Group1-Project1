"""Milestone 4: end-to-end celebrity detection and identification.

Owner: Murat Akca. Combines the Milestone 1 ResNet18 classifier with the Milestone 3 YOLOv8n detector.

1. YOLOv8n finds the faces (class-agnostic NMS, so one box per face) and scores the five identities.
2. The image is also shrunk onto a plain canvas and searched again; boxes from both scales are merged by NMS.
   Close-up portraits show a face far larger than any face in the training scenes; shrinking it brings the face
   to a familiar scale. (detection="retry_if_empty" searches the canvas only when step 1 finds nothing.)
3. Each face is cropped to the framing of the CelebA portraits the classifier was trained on, and ResNet18
   classifies the crop.
4. The two identity distributions are averaged with weight `yolo_weight`. A face whose fused confidence is
   below `unknown_threshold` is labelled "unknown": the box stays, the name is withheld.
5. If no face is found at all, ResNet18 classifies the whole image (no box).

The detection mode, canvas factor, weight and threshold were chosen on validation by evaluate.py (results/selected_config.json).

    python project_milestone04/pipeline.py photo.jpg                     # one image
    python project_milestone04/pipeline.py photo1.jpg photo2.png folder/  # several images or folders
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "project"))
from src.pipeline import IDS, make_model  # noqa: E402  (Milestone 1 class order and architecture)

DETECTOR = ROOT / "project_milestone03" / "models" / "best.pt"
CLASSIFIER = ROOT / "project" / "models" / "group1_repro_v1" / "resnet18_frozen_backbone.pt"
CONFIG = ROOT / "project_milestone04" / "results" / "selected_config.json"
UNKNOWN = "unknown"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Detector settings selected on validation in Milestone 3 (Dario's sweep). NMS is class-agnostic here because
# identity is decided after detection, so each face should keep exactly one box.
CONFIDENCE, NMS_IOU, IMAGE_SIZE = 0.50, 0.70, 640
# Portrait framing around a face box: left and top margins as fractions of the box width and height, then the
# portrait width and height in box units. Medians over all 121 source photos (YuNet face box vs the 178x218
# aligned photo); evaluate.py measures them again.
PORTRAIT = (0.51, 0.57, 2.00, 1.87)
PORTRAIT_SIZE = (178, 218)
CANVAS_COLOR = (200, 210, 214)  # BGR, a light wall colour like the Milestone 2 backgrounds
DEFAULTS = {"yolo_weight": 0.5, "unknown_threshold": 0.0, "canvas_factor": 2, "detection": "two_scales"}
DETECTION_MODES = ("retry_if_empty", "two_scales")
COLORS = [(0, 114, 178), (213, 94, 0), (0, 158, 115), (204, 121, 167), (230, 159, 0)]  # RGB, one per identity


def load_config(path=CONFIG):
    path = Path(path)
    if not path.is_file():
        print(f"Note: {path} not found; using defaults {DEFAULTS}. Run project_milestone04/evaluate.py.")
        return dict(DEFAULTS)
    selected = json.loads(path.read_text(encoding="utf-8"))
    return {key: selected[key] for key in DEFAULTS}


def portrait_crop(rgb, box):
    """Crop the portrait around a face box, mirroring the image at its borders, resized to 178x218."""
    import cv2

    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    left, top = round(x1 - PORTRAIT[0] * width), round(y1 - PORTRAIT[1] * height)
    right, bottom = round(left + PORTRAIT[2] * width), round(top + PORTRAIT[3] * height)
    pad = max(0, -left, -top, right - rgb.shape[1], bottom - rgb.shape[0])
    if pad:
        rgb = cv2.copyMakeBorder(rgb, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
    crop = rgb[top + pad:bottom + pad, left + pad:right + pad]
    return cv2.resize(crop, PORTRAIT_SIZE, interpolation=cv2.INTER_AREA)


def fuse(detector_scores, classifier_probs, yolo_weight):
    """Weighted average of the detector's normalized class scores and the classifier's softmax."""
    if detector_scores is None:  # whole-image fallback: the classifier is the only opinion
        return np.asarray(classifier_probs)
    detector_probs = np.asarray(detector_scores) / max(float(np.sum(detector_scores)), 1e-9)
    return yolo_weight * detector_probs + (1 - yolo_weight) * np.asarray(classifier_probs)


class CelebrityPipeline:
    def __init__(self, detector=DETECTOR, classifier=CLASSIFIER, config=CONFIG, device="cpu"):
        import torch
        from torchvision import transforms
        from ultralytics import YOLO

        self.device = device
        self.settings = load_config(config) if isinstance(config, (str, Path)) else {**DEFAULTS, **config}
        self.detector = YOLO(str(detector))
        names = [str(self.detector.names[index]) for index in range(len(self.detector.names))]
        if self.detector.task != "detect" or names != IDS:
            raise ValueError(f"Expected the five-identity detector; found task={self.detector.task}, names={names}")
        saved = torch.load(classifier, map_location=device, weights_only=True)
        if saved["ids"] != IDS or saved["architecture"] != "resnet18_frozen_backbone":
            raise ValueError("Expected the Milestone 1 ResNet18 checkpoint with the same five identities")
        self.classifier = make_model("resnet18_frozen_backbone", pretrained=False).to(device)
        self.classifier.load_state_dict(saved["state_dict"])
        self.classifier.eval()
        # Same test-time preprocessing as Milestone 1 (project/src/pipeline.py, make_loader).
        self.transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                             transforms.Normalize((.485, .456, .406), (.229, .224, .225))])
        # One warm-up call creates the Ultralytics predictor whose preprocessing raw_detect() reuses.
        self.detector.predict(np.zeros((64, 64, 3), np.uint8), imgsz=IMAGE_SIZE, device=device, verbose=False)

    def raw_detect(self, bgr, confidence=CONFIDENCE, nms_iou=NMS_IOU):
        """Face boxes (n, 4) in pixels and all five sigmoid class scores (n, 5).

        model.predict() keeps only the top class score of each box; fusion needs all five, so this runs the same
        preprocessing and NMS and reads the class scores of the boxes NMS kept. Boxes and confidences match
        model.predict(..., agnostic_nms=True) exactly (test_pipeline.py checks this).
        """
        import torch
        from ultralytics.utils.nms import non_max_suppression
        from ultralytics.utils.ops import scale_boxes

        predictor = self.detector.predictor
        batch = predictor.preprocess([bgr])
        with torch.no_grad():
            raw = predictor.inference(batch)
        raw = raw[0] if isinstance(raw, (list, tuple)) else raw
        kept, indices = non_max_suppression(raw, confidence, nms_iou, agnostic=True, max_det=300, return_idxs=True)
        kept, indices = kept[0], indices[0].long()
        if not len(kept):
            return np.zeros((0, 4)), np.zeros((0, len(IDS)))
        scores = raw[0, 4:4 + len(IDS), indices].T
        boxes = scale_boxes(batch.shape[2:], kept[:, :4].clone(), bgr.shape)
        return boxes.cpu().numpy(), scores.cpu().numpy()

    def classify(self, rgb_images):
        import torch
        from PIL import Image

        batch = torch.stack([self.transform(Image.fromarray(image)) for image in rgb_images]).to(self.device)
        with torch.no_grad():
            return torch.softmax(self.classifier(batch), 1).cpu().numpy()

    def detect_on_canvas(self, bgr, factor, confidence=CONFIDENCE):
        """Detect on the image shrunk by `factor` (centred on a plain canvas); boxes in original pixels."""
        height, width = bgr.shape[:2]
        canvas = np.full((int(height * factor), int(width * factor), 3), CANVAS_COLOR, np.uint8)
        top, left = (canvas.shape[0] - height) // 2, (canvas.shape[1] - width) // 2
        canvas[top:top + height, left:left + width] = bgr
        boxes, scores = self.raw_detect(canvas, confidence)
        return boxes - [left, top, left, top], scores

    def analyse(self, image, canvas_factor=None, confidence=CONFIDENCE, detection=None):
        """Run both models once. decide() turns the result into labels for any weight and threshold."""
        import torch
        from torchvision.ops import nms

        rgb = load_rgb(image)
        bgr = np.ascontiguousarray(rgb[..., ::-1])
        factor = canvas_factor or self.settings["canvas_factor"]
        detection = detection or self.settings["detection"]
        if detection not in DETECTION_MODES:
            raise ValueError(f"detection must be one of {DETECTION_MODES}")
        boxes, scores = self.raw_detect(bgr, confidence)
        stage = "detected"
        if detection == "two_scales":
            canvas_boxes, canvas_scores = self.detect_on_canvas(bgr, factor, confidence)
            boxes, scores = np.concatenate([boxes, canvas_boxes]), np.concatenate([scores, canvas_scores])
            if len(boxes):
                keep = nms(torch.as_tensor(boxes, dtype=torch.float32),
                           torch.as_tensor(scores.max(1), dtype=torch.float32), NMS_IOU).numpy()
                boxes, scores = boxes[keep], scores[keep]
            stage = "two_scales"
        elif not len(boxes):
            boxes, scores = self.detect_on_canvas(bgr, factor, confidence)
            stage = "detected_on_canvas"
        boxes, scores = drop_nested(boxes, scores)
        if len(boxes):
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, rgb.shape[1])
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, rgb.shape[0])
            probs = self.classify([portrait_crop(rgb, box) for box in boxes])
            faces = [{"box": box.tolist(), "detector_scores": score, "classifier_probs": prob}
                     for box, score, prob in zip(boxes, scores, probs)]
        else:
            stage = "whole_image"
            faces = [{"box": None, "detector_scores": None, "classifier_probs": self.classify([rgb])[0]}]
        faces.sort(key=lambda face: face["box"][0] if face["box"] else 0)  # left to right
        return {"stage": stage, "size": rgb.shape[:2], "faces": faces}

    def decide(self, analysis, yolo_weight=None, unknown_threshold=None):
        weight = self.settings["yolo_weight"] if yolo_weight is None else yolo_weight
        threshold = self.settings["unknown_threshold"] if unknown_threshold is None else unknown_threshold
        return [decide_face(face, analysis["stage"], weight, threshold) for face in analysis["faces"]]

    def __call__(self, image, yolo_weight=None, unknown_threshold=None, canvas_factor=None, confidence=CONFIDENCE):
        analysis = self.analyse(image, canvas_factor, confidence)
        return self.decide(analysis, yolo_weight, unknown_threshold), analysis


def drop_nested(boxes, scores, inside=0.8):
    """Drop a box that lies mostly inside a larger one: a face cannot contain another face.

    On close-up portraits the detector sometimes boxes an eye or a lock of hair as well as the whole face.
    """
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = []
    for i in range(len(boxes)):
        nested = False
        for j in range(len(boxes)):
            if j != i and areas[j] > areas[i]:
                overlap_w = min(boxes[i, 2], boxes[j, 2]) - max(boxes[i, 0], boxes[j, 0])
                overlap_h = min(boxes[i, 3], boxes[j, 3]) - max(boxes[i, 1], boxes[j, 1])
                if max(0, overlap_w) * max(0, overlap_h) >= inside * areas[i]:
                    nested = True
                    break
        if not nested:
            keep.append(i)
    return boxes[keep], scores[keep]


def decide_face(face, stage, yolo_weight, unknown_threshold):
    fused = fuse(face["detector_scores"], face["classifier_probs"], yolo_weight)
    best = int(np.argmax(fused))
    result = {"identity": IDS[best] if fused[best] >= unknown_threshold else UNKNOWN,
              "confidence": float(fused[best]), "box": face["box"], "stage": stage,
              "classifier_identity": IDS[int(np.argmax(face["classifier_probs"]))],
              "classifier_probability": float(np.max(face["classifier_probs"])),
              "detector_identity": "", "detector_score": None}
    if face["detector_scores"] is not None:
        result["detector_identity"] = IDS[int(np.argmax(face["detector_scores"]))]
        result["detector_score"] = float(np.max(face["detector_scores"]))
    return result


def load_rgb(image):
    """Path, PIL image or RGB array -> RGB uint8 array."""
    import cv2
    from PIL import Image

    if isinstance(image, (str, Path)):
        bgr = cv2.imread(str(image))  # the same decoder (and EXIF rotation) as Ultralytics uses for paths
        if bgr is None:
            raise ValueError(f"Cannot read image: {image}")
        return np.ascontiguousarray(bgr[..., ::-1])
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.stack([array] * 3, -1)
    return np.ascontiguousarray(array[..., :3]).astype(np.uint8)


def draw(rgb, faces):
    """Annotated RGB copy: one coloured box per identity, grey for unknown, a banner for whole-image results."""
    from PIL import Image, ImageDraw, ImageFont

    image = Image.fromarray(load_rgb(rgb))
    canvas = ImageDraw.Draw(image)
    size = max(14, round(min(image.size) / 30))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        try:
            font = ImageFont.truetype("Arial.ttf", size)
        except OSError:
            font = ImageFont.load_default()
    line = max(2, round(min(image.size) / 200))
    for face in faces:
        known = face["identity"] != UNKNOWN
        color = COLORS[IDS.index(face["identity"])] if known else (128, 128, 128)
        label = f"{face['identity']} {face['confidence']:.2f}"
        if face["box"] is None:
            label = f"Whole image (no face box): {label}"
            left, top = 0, 0
        else:
            x1, y1, x2, y2 = face["box"]
            canvas.rectangle((x1, y1, x2, y2), outline=color, width=line)
            left, top = x1, max(0, y1 - size - 6)
        text = canvas.textbbox((left + 3, top + 2), label, font=font)
        canvas.rectangle((left, top, text[2] + 3, text[3] + 3), fill=color)
        canvas.text((left + 3, top + 2), label, fill="white", font=font)
    return np.asarray(image)


def describe(faces):
    lines = []
    for face in faces:
        where = "whole image" if face["box"] is None else "box ({:.0f}, {:.0f}, {:.0f}, {:.0f})".format(*face["box"])
        detector = (f"YOLO {face['detector_identity']} {face['detector_score']:.2f}"
                    if face["detector_identity"] else "YOLO -")
        lines.append(f"  {face['identity']:>7}  fused {face['confidence']:.2f}  [{detector}, ResNet18 "
                     f"{face['classifier_identity']} {face['classifier_probability']:.2f}]  {where}")
    return lines


def expand(paths):
    for path in paths:
        if path.is_dir():
            yield from sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
        else:
            yield path


def main():
    from PIL import Image

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="+", type=Path, help="Image files or folders.")
    parser.add_argument("--output", type=Path, default=ROOT / "tmp" / "m4_pipeline",
                        help="Folder for annotated copies and predictions.json (default: tmp/m4_pipeline).")
    parser.add_argument("--unknown-threshold", type=float, help="Override the validated rejection threshold.")
    parser.add_argument("--yolo-weight", type=float, help="Override the validated fusion weight.")
    parser.add_argument("--device", default="cpu", help="cpu, mps or a CUDA index such as 0.")
    args = parser.parse_args()
    pipeline = CelebrityPipeline(device=args.device)
    print(f"Settings: {pipeline.settings}")
    args.output.mkdir(parents=True, exist_ok=True)
    report = {}
    for path in expand(args.images):
        faces, analysis = pipeline(path, args.yolo_weight, args.unknown_threshold)
        target = args.output / f"{path.stem}_pipeline.jpg"
        Image.fromarray(draw(path, faces)).save(target, quality=92)
        named = sum(face["identity"] != UNKNOWN for face in faces)
        print(f"{path}: {len(faces)} face(s), {named} named, stage {analysis['stage']} -> {target}")
        print("\n".join(describe(faces)))
        report[str(path)] = faces
    (args.output / "predictions.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Predictions saved to {args.output / 'predictions.json'}")


if __name__ == "__main__":
    main()
