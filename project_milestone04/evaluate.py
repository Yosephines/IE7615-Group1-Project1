"""Milestone 4: choose the pipeline settings on validation, then score the held-out test split.

Owner: Murat Akca. Both checkpoints stay fixed; nothing is retrained.

Validation (36 scenes, 111 faces, plus the 19 validation source portraits) selects all four settings by the
largest (correctly named faces - wrongly named faces). A withheld name ("unknown") costs a correct name but
avoids a wrong one.
1. detection: search a shrunk copy of the image only when the original finds no face ("retry_if_empty"), or
   always, merging both scales ("two_scales"); canvas_factor: how much the copy is shrunk (2, 3 or 4).
2. yolo_weight and unknown_threshold: the fusion weight and the rejection threshold.
Ties prefer retry_if_empty, the smaller factor, an even weight (0.5), then the lower threshold.

Test (36 scenes, 113 faces, plus the 17 test source portraits) is scored once with the selected settings and
compared with each model on its own. The TA's real-photo grid shows the effect on faces of unknown people.

    python project_milestone04/evaluate.py                          # writes project_milestone04/results/
    python project_milestone04/evaluate.py --output tmp/m4_reproduce  # reproduce without touching results/
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

import pipeline as m4
from pipeline import IDS, UNKNOWN

ROOT = m4.ROOT
DATASET = ROOT / "project_milestone02" / "dataset"
PHOTOS = ROOT / "data" / "identities"
TA_IMAGE = ROOT / "project_milestone03" / "results" / "ta_image_check" / "ta_test_image.png"
# Tiles of the TA grid that are copies of our training photos (project_milestone03/results/ta_image_check);
# the other six show people outside our five identities. Keys are (row, column).
TA_KNOWN_TILES = {(0, 0): "2336", (0, 2): "2970", (2, 2): "7007"}
MATCH_IOU = 0.50
CANVAS_FACTORS = (2, 3, 4)
WEIGHTS = [round(value, 1) for value in np.arange(0, 1.01, 0.1)]
THRESHOLDS = [0.0] + [round(value, 2) for value in np.arange(0.30, 0.901, 0.01)]
EXPECTED_SHA256 = {  # from the Milestone 1 run record and the Milestone 3 README
    "classifier": "e3a8199480649cf8a63b64c0be0491bd4efa45d3d46fa6bab63757b63d8693b3",
    "detector": "64f5686b5a7760fee18cf215b1e29f4aa5af95dc90fbf0b571635ed08aa1d2e6",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def box_iou(a, b):
    inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------- data

def scenes(split):
    """[(image path, [(identity, pixel box), ...])] for one Milestone 2 split."""
    from PIL import Image

    items = []
    for image in sorted((DATASET / split / "images").glob("*.jpg")):
        width, height = Image.open(image).size
        truth = []
        for line in (DATASET / split / "labels" / f"{image.stem}.txt").read_text().splitlines():
            class_id, x, y, w, h = (float(value) for value in line.split())
            truth.append((IDS[int(class_id)], [(x - w / 2) * width, (y - h / 2) * height,
                                               (x + w / 2) * width, (y + h / 2) * height]))
        items.append((image, truth))
    return items


def portraits(split):
    """[(photo path, identity)] for the source photos assigned to one split in Milestone 2."""
    with (DATASET / "source_split.csv").open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["split"] == split]
    return [(PHOTOS / row["file"], row["identity"]) for row in rows]


# ---------------------------------------------------------------- scoring

def score_scene(truth, faces):
    """Match boxes to reference faces (IoU >= 0.5), then compare names.

    Pairs with the right name are matched first, as in Milestone 3, so a duplicate box with another name on an
    already correctly named face counts as an extra named box rather than replacing the correct one.
    """
    candidates = sorted(((f["identity"] == t[0], box_iou(t[1], f["box"]), ti, fi) for ti, t in enumerate(truth)
                         for fi, f in enumerate(faces) if f["box"] is not None), reverse=True)
    pairs, used_t, used_f = [], set(), set()
    for _, overlap, ti, fi in candidates:
        if overlap >= MATCH_IOU and ti not in used_t and fi not in used_f:
            pairs.append((ti, fi, overlap))
            used_t.add(ti)
            used_f.add(fi)
    rows = []
    for ti, fi, overlap in pairs:
        rows.append({"true_identity": truth[ti][0], "outcome": outcome(truth[ti][0], faces[fi]["identity"]),
                     "iou": round(overlap, 3), **faces[fi]})
    rows += [{"true_identity": t[0], "outcome": "missed", "iou": 0.0, "identity": "", "confidence": 0.0,
              "box": t[1]} for ti, t in enumerate(truth) if ti not in used_t]
    rows += [{"true_identity": "", "outcome": "extra_named" if f["identity"] != UNKNOWN else "extra_unknown",
              "iou": 0.0, **f} for fi, f in enumerate(faces) if fi not in used_f]
    return rows


def score_portrait(identity, faces, size=None):
    """A portrait has one person: score the largest box, or the whole-image answer.

    `iou` compares the box with where the face sits in an aligned CelebA photo, to show whether the box found
    the face at all.
    """
    main = max(faces, key=lambda f: 0 if f["box"] is None else (f["box"][2] - f["box"][0]) * (f["box"][3] - f["box"][1]))
    overlap = "" if main["box"] is None or size is None else round(box_iou(aligned_face_box(*size), main["box"]), 3)
    return [{"true_identity": identity, "outcome": outcome(identity, main["identity"]), "iou": overlap, **main}]


def aligned_face_box(height, width):
    """Expected face box in an aligned portrait, the inverse of the pipeline's PORTRAIT framing."""
    left, top, portrait_w, portrait_h = m4.PORTRAIT
    face_w, face_h = width / portrait_w, height / portrait_h
    return [left * face_w, top * face_h, left * face_w + face_w, top * face_h + face_h]


def outcome(truth, predicted):
    return "correct" if predicted == truth else "unknown" if predicted == UNKNOWN else "wrong"


def summarize(rows):
    count = {key: sum(row["outcome"] == key for row in rows)
             for key in ("correct", "wrong", "unknown", "missed", "extra_named", "extra_unknown")}
    faces = sum(row["outcome"] in ("correct", "wrong", "unknown", "missed") for row in rows)
    named_wrong = count["wrong"] + count["extra_named"]
    named = count["correct"] + named_wrong
    return {"faces": faces, **count, "accuracy": count["correct"] / faces if faces else 0.0,
            "precision": count["correct"] / named if named else 0.0, "net": count["correct"] - named_wrong}


def decided_rows(pipeline, analysed, weight, threshold):
    rows = []
    for item in analysed:
        faces = pipeline.decide(item["analysis"], weight, threshold)
        scored = (score_scene(item["truth"], faces) if item["kind"] == "scene"
                  else score_portrait(item["truth"], faces, item["analysis"]["size"]))
        rows += [{"image": item["name"], "kind": item["kind"], "stage": item["analysis"]["stage"], **row}
                 for row in scored]
    return rows


def analyse_split(pipeline, split, canvas_factor, detection):
    items = [{"kind": "scene", "name": path.relative_to(DATASET).as_posix(), "truth": truth,
              "analysis": pipeline.analyse(path, canvas_factor, detection=detection)} for path, truth in scenes(split)]
    items += [{"kind": "portrait", "name": path.relative_to(ROOT).as_posix(), "truth": identity,
               "analysis": pipeline.analyse(path, canvas_factor, detection=detection)}
              for path, identity in portraits(split)]
    return items


# ---------------------------------------------------------------- baselines

def detector_alone(pipeline, split):
    """Milestone 3 as reported: YOLOv8n with class-aware NMS and no canvas retry."""
    rows = []
    for path, truth in scenes(split):
        faces = detector_faces(pipeline, path)
        rows += [{"image": path.name, "kind": "scene", **row} for row in score_scene(truth, faces)]
    for path, identity in portraits(split):
        faces = detector_faces(pipeline, path)
        if faces:
            scored = score_portrait(identity, faces)
        else:
            scored = [{"true_identity": identity, "outcome": "missed", "iou": "", "identity": "", "box": None}]
        rows += [{"image": path.name, "kind": "portrait", **row} for row in scored]
    return rows


def detector_faces(pipeline, path):
    result = pipeline.detector.predict(str(path), conf=m4.CONFIDENCE, iou=m4.NMS_IOU, agnostic_nms=False,
                                       imgsz=m4.IMAGE_SIZE, device=pipeline.device, verbose=False)[0]
    return [{"identity": IDS[int(label)], "confidence": float(score), "box": box}
            for box, score, label in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(),
                                         result.boxes.cls.tolist())]


def classifier_alone(pipeline, analysed):
    """Milestone 1 as reported: ResNet18 on the whole portrait; on scenes it classifies the detector's crops."""
    rows = decided_rows(pipeline, [item for item in analysed if item["kind"] == "scene"], 0.0, 0.0)
    for item in analysed:
        if item["kind"] == "portrait":
            probs = pipeline.classify([m4.load_rgb(ROOT / item["name"])])[0]
            face = {"identity": IDS[int(np.argmax(probs))], "confidence": float(np.max(probs)), "box": None}
            rows += [{"image": item["name"], "kind": "portrait", **row}
                     for row in score_portrait(item["truth"], [face])]
    return rows


# ---------------------------------------------------------------- selection

def choose_settings(pipeline):
    """Best fusion settings for every detection mode and canvas factor; returns the overall winner."""
    rows, runs = [], []
    for detection in m4.DETECTION_MODES:
        for factor in CANVAS_FACTORS:
            items = analyse_split(pipeline, "val", factor, detection)
            best, grid = choose_fusion(pipeline, items)
            rows.append({"detection": detection, "canvas_factor": factor, "yolo_weight": best["yolo_weight"],
                         "unknown_threshold": best["unknown_threshold"],
                         **{key: best[key] for key in ("net", "correct", "wrong", "extra_named", "unknown", "missed")}})
            runs.append((best, grid, items))
    pick = max(range(len(rows)), key=lambda i: (rows[i]["net"], rows[i]["detection"] == "retry_if_empty",
                                                 -rows[i]["canvas_factor"]))
    return rows[pick], runs[pick], rows


def choose_fusion(pipeline, analysed):
    rows = []
    for weight in WEIGHTS:
        for threshold in THRESHOLDS:
            summary = summarize(decided_rows(pipeline, analysed, weight, threshold))
            rows.append({"yolo_weight": weight, "unknown_threshold": threshold, **summary})
    best = max(rows, key=lambda row: (row["net"], -abs(row["yolo_weight"] - 0.5), -row["unknown_threshold"]))
    return best, rows


def threshold_curve(pipeline, splits, weight):
    """Accuracy and precision against the unknown threshold, for the analysis section (not used for selection)."""
    rows = []
    for split, items in splits.items():
        for threshold in THRESHOLDS:
            summary = summarize(decided_rows(pipeline, items, weight, threshold))
            rows.append({"split": split, "unknown_threshold": threshold, **summary})
    return rows


def plot_threshold_curve(rows, selected, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3.8))
    for split, color in (("val", "#0072b2"), ("test", "#d55e00")):
        subset = [row for row in rows if row["split"] == split and row["unknown_threshold"] > 0]
        thresholds = [row["unknown_threshold"] for row in subset]
        ax.plot(thresholds, [row["accuracy"] for row in subset], color=color, label=f"{split} accuracy")
        ax.plot(thresholds, [row["precision"] for row in subset], color=color, linestyle="--",
                label=f"{split} precision")
    ax.axvline(selected, color="#555555", linewidth=1)
    ax.text(selected + 0.005, 0.42, f"selected on val: {selected:.2f}", color="#555555")
    ax.set(xlabel="Unknown threshold (fused confidence)", ylabel="Share of faces",
           title="Withholding low-confidence names: fewer wrong names, fewer correct names", ylim=(0.4, 1.01))
    ax.grid(alpha=.2)
    ax.legend(ncol=2, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def measure_portrait_framing():
    """Median portrait framing around YuNet face boxes on all source photos (the PORTRAIT constant)."""
    sys.path.insert(0, str(ROOT / "project_milestone02"))
    import m2_dataset
    from PIL import Image

    detector = m2_dataset.load_detector()
    ratios = []
    for photo in sorted(PHOTOS.glob("*/*.jpg")):
        rgb = np.asarray(Image.open(photo).convert("RGB"))
        height, width = rgb.shape[:2]
        faces = m2_dataset.detect_faces(detector, rgb)
        x1, y1, x2, y2, _ = min(faces, key=lambda f: ((f[0] + f[2]) / 2 - width / 2) ** 2
                                + ((f[1] + f[3]) / 2 - height / 2) ** 2)
        ratios.append([x1 / (x2 - x1), y1 / (y2 - y1), width / (x2 - x1), height / (y2 - y1)])
    median = np.median(ratios, axis=0)
    return {"photos": len(ratios), "median": [round(float(v), 3) for v in median],
            "std": [round(float(v), 3) for v in np.std(ratios, axis=0)], "used": list(m4.PORTRAIT)}


# ---------------------------------------------------------------- report

def ta_check(pipeline, output):
    from PIL import Image

    faces, analysis = pipeline(TA_IMAGE)
    height, width = analysis["size"]
    rows = []
    for face in faces:
        cx, cy = (face["box"][0] + face["box"][2]) / 2, (face["box"][1] + face["box"][3]) / 2
        tile = (int(cy * 3 // height), int(cx * 3 // width))
        truth = TA_KNOWN_TILES.get(tile, "not one of the five")
        verdict = ("correct" if face["identity"] == truth else "withheld (unknown person)"
                   if face["identity"] == UNKNOWN and truth not in IDS else "withheld (known person)"
                   if face["identity"] == UNKNOWN else "wrong name")
        rows.append({"tile_row": tile[0], "tile_column": tile[1], "truth": truth, "identity": face["identity"],
                     "confidence": round(face["confidence"], 3), "detector": face["detector_identity"],
                     "detector_score": round(face["detector_score"], 3), "classifier": face["classifier_identity"],
                     "classifier_probability": round(face["classifier_probability"], 3), "verdict": verdict})
    rows.sort(key=lambda row: (row["tile_row"], row["tile_column"]))
    write_csv(output / "ta_check.csv", rows)
    Image.fromarray(m4.draw(TA_IMAGE, faces)).save(output / "ta_check.jpg", quality=92)
    return rows


def table(rows, columns):
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(" --- " if i == 0 else " ---: " for i in range(len(columns))) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(f"{row[c]:.3f}" if isinstance(row[c], float) else str(row[c]) for c in columns) + " |")
    return "\n".join(lines)


def per_identity(rows):
    result = []
    for identity in IDS + ["All"]:
        subset = [row for row in rows if identity == "All" or row["true_identity"] == identity
                  or (row["outcome"] == "extra_named" and row["identity"] == identity)]
        summary = summarize(subset)
        result.append({"identity": identity, "faces": summary["faces"], "correct": summary["correct"],
                       "wrong": summary["wrong"], "unknown": summary["unknown"], "missed": summary["missed"],
                       "accuracy": summary["accuracy"]})
    return result


def write_summary(output, config, detection_rows, metrics, identities, ta_rows, framing):
    selected = (f"detection = {config['detection']}, canvas_factor = {config['canvas_factor']}, "
                f"yolo_weight = {config['yolo_weight']}, unknown_threshold = {config['unknown_threshold']}")
    val = config["validation"]
    named_unknown = sum(row["truth"] not in IDS and row["identity"] != UNKNOWN for row in ta_rows)
    unknown_faces = sum(row["truth"] not in IDS for row in ta_rows)
    text = f"""# Milestone 4 pipeline results

Generated by `project_milestone04/evaluate.py`. Settings were chosen on validation only; test was scored once.

## Selected settings (validation)

{selected}

Rule: {config['rule']}

On validation ({val['faces']} faces: 111 in scenes, 19 portraits) these settings name {val['correct']} faces
correctly, {val['wrong'] + val['extra_named']} wrongly and withhold {val['unknown']} as unknown
(no-rejection setting: {config['validation_without_rejection']['correct']} correct,
{config['validation_without_rejection']['wrong'] + config['validation_without_rejection']['extra_named']} wrong).

Best fusion settings for each detection mode and canvas factor (validation, net = correct - wrong names):

{table(detection_rows, ["detection", "canvas_factor", "yolo_weight", "unknown_threshold", "net", "correct", "wrong", "unknown", "missed"])}

## Held-out test

Accuracy = correct / faces. Precision = correct / faces given a name (wrong names include named boxes on
no face). "Unknown" faces are boxed but not named. Scenes: 36 images, 113 faces. Portraits: 17 source photos.
"YOLOv8n alone" runs `model.predict` image by image with the Milestone 3 settings (class-aware NMS); the
Milestone 3 report used batched `model.val`, which keeps one fewer low-confidence box on test_0018.

{table(metrics, ["method", "set", "faces", "correct", "wrong", "unknown", "missed", "extra_named", "accuracy", "precision"])}

## Final pipeline per identity (test scenes and portraits)

{table(identities, ["identity", "faces", "correct", "wrong", "unknown", "missed", "accuracy"])}

## TA real-photo grid

{named_unknown} of {unknown_faces} unknown people still receive a celebrity name (the Milestone 3 detector named
all {unknown_faces}).

{table(ta_rows, ["tile_row", "tile_column", "truth", "identity", "confidence", "detector", "detector_score", "classifier", "classifier_probability", "verdict"])}

## Portrait framing check

Median over {framing.get('photos', 0)} source photos (left, top, width, height in face-box units):
{framing.get('median', 'not measured')}; used: {framing.get('used')}.
"""
    (output / "summary.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=ROOT / "project_milestone04" / "results")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-framing", action="store_true", help="Skip the YuNet portrait-framing check.")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    checksums = {"classifier": sha256(m4.CLASSIFIER), "detector": sha256(m4.DETECTOR)}
    for name, digest in checksums.items():
        if digest != EXPECTED_SHA256[name]:
            print(f"Warning: the {name} checkpoint differs from the submitted one ({digest}).")
    pipeline = m4.CelebrityPipeline(config=m4.DEFAULTS, device=args.device)

    print("Validation: choosing the detection mode, canvas factor, fusion weight and unknown threshold ...")
    chosen, (best, grid, val_items), detection_rows = choose_settings(pipeline)
    detection, canvas_factor = chosen["detection"], chosen["canvas_factor"]
    write_csv(args.output / "validation_detection.csv", detection_rows)
    write_csv(args.output / "validation_grid.csv", grid)
    no_rejection = max((row for row in grid if row["unknown_threshold"] == 0.0 and row["yolo_weight"] == best["yolo_weight"]),
                       key=lambda row: row["net"])
    config = {"yolo_weight": float(best["yolo_weight"]), "unknown_threshold": float(best["unknown_threshold"]),
              "canvas_factor": canvas_factor, "detection": detection, "selection_split": "val",
              "rule": "maximum (correct names - wrong names) on validation scenes and portraits; ties prefer "
                      "retry_if_empty, the smaller canvas factor, yolo_weight nearest 0.5, then the lower threshold",
              "detector": {"confidence": m4.CONFIDENCE, "nms_iou": m4.NMS_IOU, "agnostic_nms": True,
                           "image_size": m4.IMAGE_SIZE},
              "validation": {key: best[key] for key in ("faces", "correct", "wrong", "unknown", "missed",
                                                         "extra_named", "accuracy", "precision")},
              "validation_without_rejection": {key: no_rejection[key] for key in ("correct", "wrong", "extra_named")},
              "checkpoints_sha256": checksums}
    (args.output / "selected_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"Selected {detection=}, {canvas_factor=}, yolo_weight={config['yolo_weight']}, "
          f"unknown_threshold={config['unknown_threshold']}")

    print("Test: scoring once with the selected settings ...")
    pipeline.settings.update({key: config[key] for key in m4.DEFAULTS})
    test_items = analyse_split(pipeline, "test", canvas_factor, detection)
    final_rows = decided_rows(pipeline, test_items, config["yolo_weight"], config["unknown_threshold"])
    methods = {
        "YOLOv8n alone (Milestone 3)": detector_alone(pipeline, "test"),
        "ResNet18 alone (Milestone 1)": classifier_alone(pipeline, test_items),
        "Pipeline, no rejection": decided_rows(pipeline, test_items, config["yolo_weight"], 0.0),
        "Pipeline (final)": final_rows,
    }
    metrics = []
    for method, rows in methods.items():
        for kind in ("scene", "portrait"):
            summary = summarize([row for row in rows if row["kind"] == kind])
            metrics.append({"method": method, "set": "scenes" if kind == "scene" else "portraits", **summary})
    write_csv(args.output / "test_metrics.csv", metrics)
    curve = threshold_curve(pipeline, {"val": val_items, "test": test_items}, config["yolo_weight"])
    write_csv(args.output / "threshold_curve.csv", curve)
    plot_threshold_curve(curve, config["unknown_threshold"], args.output / "threshold_curve.png")
    identities = per_identity(final_rows)
    write_csv(args.output / "test_per_identity.csv", identities)

    prediction_rows = []
    for split, rows in (("val", decided_rows(pipeline, val_items, config["yolo_weight"], config["unknown_threshold"])),
                        ("test", final_rows)):
        for row in rows:
            box = row.get("box")
            prediction_rows.append({
                "split": split, "kind": row["kind"], "image": row["image"], "stage": row.get("stage", ""),
                "true_identity": row["true_identity"], "predicted_identity": row.get("identity", ""),
                "outcome": row["outcome"], "confidence": round(row.get("confidence", 0.0), 4),
                "detector_identity": row.get("detector_identity", ""),
                "detector_score": "" if row.get("detector_score") is None else round(row["detector_score"], 4),
                "classifier_identity": row.get("classifier_identity", ""),
                "classifier_probability": round(row.get("classifier_probability", 0.0), 4),
                "iou": row["iou"], "box": "" if box is None else " ".join(f"{v:.1f}" for v in box)})
    write_csv(args.output / "predictions.csv", prediction_rows)

    print("TA real-photo grid ...")
    ta_rows = ta_check(pipeline, args.output)
    framing = {}
    if not args.skip_framing:
        try:
            framing = measure_portrait_framing()
            (args.output / "portrait_framing.json").write_text(json.dumps(framing, indent=2) + "\n", encoding="utf-8")
        except Exception as error:  # YuNet download or OpenCV problems should not block the metrics
            print(f"Portrait-framing check skipped: {error}")
    write_summary(args.output, config, detection_rows, metrics, identities, ta_rows, framing)
    print((args.output / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
