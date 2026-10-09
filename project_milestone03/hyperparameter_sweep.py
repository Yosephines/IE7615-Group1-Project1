"""Compare inference thresholds on validation and build a held-out detection gallery.

Owner: Dario Garza. The checkpoint is held fixed; this script never retrains it.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import platform
import tempfile
import textwrap
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IDENTITIES = ["7007", "2970", "2336", "7", "4428"]
AP_CONFIDENCE_FLOOR = 0.001
MATCH_IOU = 0.50
BASELINE = {"confidence": 0.25, "nms_iou": 0.70, "agnostic_nms": False}
ULTRALYTICS_VERSION = "8.4.174"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def box_iou(first, second):
    intersection = max(0.0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0.0, min(first[3], second[3]) - max(first[1], second[1])
    )
    area_first = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_second = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = area_first + area_second - intersection
    return intersection / union if union else 0.0


def read_labels(path, predictions=False):
    path = Path(path)
    if not path.is_file():
        if predictions:
            # Ultralytics creates no prediction file when an image has no detections.
            return []
        raise FileNotFoundError(f"Ground-truth label is missing: {path}")
    boxes = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        values = [float(value) for value in line.split()]
        expected = 6 if predictions else 5
        if len(values) != expected or not all(math.isfinite(value) for value in values):
            raise ValueError(f"Invalid YOLO row at {path}:{line_number}")
        class_id, center_x, center_y, width, height = values[:5]
        if class_id != int(class_id) or not 0 <= class_id < len(IDENTITIES):
            raise ValueError(f"Invalid identity at {path}:{line_number}")
        box = [center_x - width / 2, center_y - height / 2,
               center_x + width / 2, center_y + height / 2]
        if width <= 0 or height <= 0:
            raise ValueError(f"Non-positive box dimensions at {path}:{line_number}")
        if not predictions and (min(box) < -1e-5 or max(box) > 1.00001):
            raise ValueError(f"Ground-truth box is outside the image at {path}:{line_number}")
        confidence = values[5] if predictions else 1.0
        if not 0 <= confidence <= 1:
            raise ValueError(f"Invalid confidence at {path}:{line_number}")
        boxes.append({"class_id": int(class_id), "box": box, "confidence": confidence})
    return boxes


def match_detections(ground_truth, predictions, match_iou=MATCH_IOU):
    """Match once per box, in descending IoU, using the baseline report's method."""
    candidates = []
    for truth_index, truth in enumerate(ground_truth):
        for prediction_index, prediction in enumerate(predictions):
            overlap = box_iou(truth["box"], prediction["box"])
            if truth["class_id"] == prediction["class_id"] and overlap >= match_iou:
                candidates.append((overlap, truth_index, prediction_index))
    used_truth, used_predictions, matches = set(), set(), []
    for overlap, truth_index, prediction_index in sorted(candidates, reverse=True):
        if truth_index not in used_truth and prediction_index not in used_predictions:
            used_truth.add(truth_index)
            used_predictions.add(prediction_index)
            matches.append((truth_index, prediction_index, overlap))
    return matches, used_truth, used_predictions


def summarize_counts(counts):
    true_positives, false_positives, false_negatives = (counts[key] for key in ("tp", "fp", "fn"))
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0.0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives else 0.0
    return {"tp": true_positives, "fp": false_positives, "fn": false_negatives,
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "mean_matched_iou": sum(counts["ious"]) / len(counts["ious"]) if counts["ious"] else None}


def evaluate_saved_predictions(dataset, prediction_dir, split, confidence):
    counts = {class_id: {"tp": 0, "fp": 0, "fn": 0, "ious": []} for class_id in range(len(IDENTITIES))}
    image_records = []
    for image_path in sorted((Path(dataset) / split / "images").glob("*.jpg")):
        ground_truth = read_labels(Path(dataset) / split / "labels" / f"{image_path.stem}.txt")
        predictions = [box for box in read_labels(Path(prediction_dir) / f"{image_path.stem}.txt", True)
                       if box["confidence"] >= confidence]
        matches, used_truth, used_predictions = match_detections(ground_truth, predictions)
        for truth_index, _, overlap in matches:
            class_counts = counts[ground_truth[truth_index]["class_id"]]
            class_counts["tp"] += 1
            class_counts["ious"].append(overlap)
        for truth_index, truth in enumerate(ground_truth):
            if truth_index not in used_truth:
                counts[truth["class_id"]]["fn"] += 1
        for prediction_index, prediction in enumerate(predictions):
            if prediction_index not in used_predictions:
                counts[prediction["class_id"]]["fp"] += 1
        confused_truth = {truth_index for truth_index, truth in enumerate(ground_truth)
                          if truth_index not in used_truth and any(
                              prediction["class_id"] != truth["class_id"]
                              and box_iou(truth["box"], prediction["box"]) >= MATCH_IOU
                              for prediction in predictions)}
        conflicting_pairs = sum(
            first["class_id"] != second["class_id"] and box_iou(first["box"], second["box"]) >= MATCH_IOU
            for first, second in itertools.combinations(predictions, 2)
        )
        image_records.append({"image": image_path.name, "ground_truth": ground_truth,
                              "predictions": predictions, "matches": matches,
                              "tp": len(matches), "fp": len(predictions) - len(matches),
                              "fn": len(ground_truth) - len(matches),
                              "identity_confusions": len(confused_truth),
                              "conflicting_prediction_pairs": conflicting_pairs})
    total = {key: sum(value[key] for value in counts.values()) for key in ("tp", "fp", "fn")}
    total["ious"] = [overlap for value in counts.values() for overlap in value["ious"]]
    overall = summarize_counts(total)
    overall["identity_confusions"] = sum(record["identity_confusions"] for record in image_records)
    overall["conflicting_prediction_pairs"] = sum(record["conflicting_prediction_pairs"] for record in image_records)
    return {"overall": overall, "per_class": {IDENTITIES[class_id]: summarize_counts(value)
            for class_id, value in counts.items()}, "images": image_records}


def audit_dataset(dataset):
    """Check file pairs, labels, class coverage and source-photo separation."""
    dataset = Path(dataset).resolve()
    import yaml
    specification = yaml.safe_load((dataset / "data.yaml").read_text(encoding="utf-8"))
    names = specification["names"]
    ordered_names = [str(names[index]) for index in range(len(names))] if isinstance(names, dict) else list(map(str, names))
    if ordered_names != IDENTITIES:
        raise ValueError(f"Dataset class order must be {IDENTITIES}; found {ordered_names}")
    digest, summary = hashlib.sha256(), {}
    for split in ("train", "val", "test"):
        images = sorted((dataset / split / "images").glob("*.jpg"))
        labels = sorted((dataset / split / "labels").glob("*.txt"))
        if not images or {image.stem for image in images} != {label.stem for label in labels}:
            raise ValueError(f"Missing image/label pairs in {split}")
        class_counts = {identity: 0 for identity in IDENTITIES}
        for label in labels:
            for box in read_labels(label):
                class_counts[IDENTITIES[box["class_id"]]] += 1
        if not all(class_counts.values()):
            raise ValueError(f"An identity has no labels in {split}")
        summary[split] = {"images": len(images), "faces": sum(class_counts.values()), "faces_per_identity": class_counts}
        for path in sorted(images + labels):
            digest.update(path.relative_to(dataset).as_posix().encode())
            content = path.read_bytes()
            digest.update(content.replace(b"\r\n", b"\n") if path.suffix == ".txt" else content)
    source_sets = {split: set() for split in summary}
    with (dataset / "source_split.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            source_sets[row["split"]].add(row["sha256"])
            source_path = dataset.parents[1] / "data" / "identities" / row["file"]
            if not source_path.is_file() or sha256(source_path) != row["sha256"]:
                raise ValueError(f"Missing or changed source photo: {row['file']}")
    for first, second in itertools.combinations(source_sets, 2):
        if source_sets[first] & source_sets[second]:
            raise ValueError(f"Source photos are shared between {first} and {second}")
    with (dataset / "manifest.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            source_file = dataset.parents[1] / "data" / "identities" / row["source_file"]
            if not source_file.is_file() or sha256(source_file) not in source_sets[row["split"]]:
                raise ValueError(f"Scene contains a source photo from another split: {row['image']}")
    return {"splits": summary, "dataset_sha256": digest.hexdigest(), "shared_source_photos": 0}


def resolve_checkpoint(checkpoint, training_zip, work_dir):
    if bool(checkpoint) == bool(training_zip):
        raise ValueError("Provide exactly one of --checkpoint or --training-zip.")
    if checkpoint:
        checkpoint = Path(checkpoint).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Trained checkpoint is missing: {checkpoint}. Supply Aditi's best.pt.")
        return checkpoint
    archive_path = Path(training_zip).resolve()
    with zipfile.ZipFile(archive_path) as archive:
        candidates = [name for name in archive.namelist() if name == "best.pt" or name.endswith("/best.pt")]
        if len(candidates) != 1:
            raise ValueError(f"Expected one best.pt inside the training ZIP; found {len(candidates)}.")
        # Read only the checkpoint bytes; no paths or code from the ZIP are extracted.
        target = Path(work_dir) / "best.pt"
        target.write_bytes(archive.read(candidates[0]))
    return target


def config_id(configuration):
    mode = "agnostic" if configuration["agnostic_nms"] else "aware"
    return f"conf_{configuration['confidence']:.2f}_iou_{configuration['nms_iou']:.2f}_{mode}"


def build_grid(confidences, nms_ious, agnostic_modes):
    for value in list(confidences) + list(nms_ious):
        if not math.isfinite(value) or not 0 < value < 1:
            raise ValueError("Confidence and NMS IoU values must be finite and between 0 and 1.")
    if min(confidences) < AP_CONFIDENCE_FLOOR:
        raise ValueError("Sweep confidence cannot be lower than the AP confidence floor.")
    configurations = [{"confidence": confidence, "nms_iou": overlap, "agnostic_nms": mode}
                      for confidence, overlap, mode in itertools.product(
                          sorted(set(confidences)), sorted(set(nms_ious)), sorted(set(agnostic_modes)))]
    if BASELINE not in configurations:
        configurations.append(BASELINE.copy())
    return configurations


def choose_configuration(validation_rows):
    """Maximize validation micro-F1; break ties using recall, precision, AP and baseline proximity."""
    if not validation_rows or any(row["split"] != "val" for row in validation_rows):
        raise ValueError("Configuration selection must use validation rows only.")
    def ranking(row):
        distance = abs(row["confidence"] - BASELINE["confidence"]) + abs(row["nms_iou"] - BASELINE["nms_iou"])
        return (-round(row["f1"], 12), -round(row["recall"], 12), -round(row["precision"], 12),
                -round(row["map50_95"], 12), row["agnostic_nms"], distance, row["config_id"])
    return min(validation_rows, key=ranking)


def validate_model(model, dataset_yaml, output_dir, split, nms_iou, agnostic_nms, device, batch, image_size):
    run_name = f"{split}_iou_{nms_iou:.2f}_{'agnostic' if agnostic_nms else 'aware'}"
    metrics = model.val(data=str(dataset_yaml), split=split, imgsz=image_size, batch=batch,
                        device=device, workers=0, conf=AP_CONFIDENCE_FLOOR, iou=nms_iou,
                        agnostic_nms=agnostic_nms, rect=True, augment=False, plots=False,
                        save_txt=True, save_conf=True, project=str(output_dir / "raw_predictions"),
                        name=run_name, exist_ok=False, verbose=False)
    per_class = {}
    for index, class_id in enumerate(metrics.box.ap_class_index):
        precision, recall, ap50, ap50_95 = metrics.box.class_result(index)
        per_class[IDENTITIES[int(class_id)]] = {"map50": float(ap50), "map50_95": float(ap50_95),
                                             "precision_auto": float(precision), "recall_auto": float(recall)}
    standard = {"map50": float(metrics.box.map50), "map50_95": float(metrics.box.map),
                "precision_auto": float(metrics.box.mp), "recall_auto": float(metrics.box.mr),
                "per_class": per_class}
    return Path(metrics.save_dir) / "labels", standard


def plot_sweep(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    modes = sorted({row["agnostic_nms"] for row in rows})
    confidences = sorted({row["confidence"] for row in rows})
    overlaps = sorted({row["nms_iou"] for row in rows})
    figure, axes = plt.subplots(1, len(modes), figsize=(6 * len(modes), 4), squeeze=False, constrained_layout=True)
    for axis, mode in zip(axes[0], modes):
        lookup = {(row["confidence"], row["nms_iou"]): row["f1"] for row in rows if row["agnostic_nms"] == mode}
        matrix = np.array([[lookup.get((confidence, overlap), np.nan) for overlap in overlaps] for confidence in confidences])
        image = axis.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
        axis.set_xticks(range(len(overlaps)), [f"{value:.2f}" for value in overlaps])
        axis.set_yticks(range(len(confidences)), [f"{value:.2f}" for value in confidences])
        axis.set(xlabel="NMS IoU threshold", ylabel="Confidence threshold",
                 title=f"Validation micro-F1 | {'class-agnostic' if mode else 'class-aware'} NMS")
        for row_index in range(len(confidences)):
            for column_index in range(len(overlaps)):
                value = matrix[row_index, column_index]
                if not np.isnan(value):
                    axis.text(column_index, row_index, f"{value:.3f}", ha="center", va="center",
                              color="white" if value > 0.6 else "black")
        figure.colorbar(image, ax=axis, label="F1")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def render_gallery_panel(dataset, record, configuration, target, source):
    from PIL import Image, ImageDraw, ImageFont
    original = Image.open(Path(dataset) / "test" / "images" / record["image"]).convert("RGB")
    width, height = original.size
    canvas = Image.new("RGB", (width * 2 + 24, height + 150), "#f2f5f8")
    canvas.paste(original, (8, 58))
    canvas.paste(original, (width + 16, 58))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
        small_font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
        small_font = font
    draw.text((8, 7), f"{record['image']} | {source} | confidence={configuration['confidence']:.2f}, "
              f"NMS IoU={configuration['nms_iou']:.2f}, agnostic={configuration['agnostic_nms']}", fill="#203244", font=font)
    draw.text((8, 32), "Ground truth: identity IDs", fill="#203244", font=small_font)
    draw.text((width + 16, 32), "Predictions: green = correct; red = unmatched", fill="#203244", font=small_font)
    correct_predictions = {match[1] for match in record["matches"]}
    for is_prediction, boxes, offset in [(False, record["ground_truth"], 8), (True, record["predictions"], width + 16)]:
        for index, box in enumerate(boxes):
            coordinates = [box["box"][0] * width + offset, box["box"][1] * height + 58,
                           box["box"][2] * width + offset, box["box"][3] * height + 58]
            color = ("#16834a" if index in correct_predictions else "#bd2533") if is_prediction else "#1559b5"
            label = IDENTITIES[box["class_id"]] + (f" {box['confidence']:.2f}" if is_prediction else "")
            draw.rectangle(coordinates, outline=color, width=3)
            left, top = coordinates[0], max(58, coordinates[1] - 23)
            text_box = draw.textbbox((left + 3, top), label, font=small_font)
            draw.rectangle((left, top, text_box[2] + 4, top + 23), fill=color)
            draw.text((left + 3, top), label, fill="white", font=small_font)
    missed_ids = [IDENTITIES[box["class_id"]] for index, box in enumerate(record["ground_truth"])
                  if index not in {match[0] for match in record["matches"]}]
    status = "Success" if record["fp"] == record["fn"] == 0 else "Failure"
    caption = (f"{status}: {record['tp']}/{len(record['ground_truth'])} faces correctly detected and identified; "
               f"{record['fp']} false positives; {record['fn']} misses. "
               f"Identity confusions: {record['identity_confusions']}; overlapping different-ID pairs: "
               f"{record['conflicting_prediction_pairs']}.")
    if missed_ids:
        caption += f" Unmatched reference IDs: {', '.join(missed_ids)}."
    for line_index, line in enumerate(textwrap.wrap(caption, width=180)):
        draw.text((8, height + 66 + line_index * 22), line, fill="#203244", font=small_font)
    canvas.save(target, quality=90)
    return status, caption


def create_gallery(dataset, output_dir, selected, baseline, count):
    selected_cases = [(record, selected["configuration"], "selected configuration") for record in selected["fixed"]["images"]
                      if len(record["ground_truth"]) >= 2]
    baseline_cases = [(record, baseline["configuration"], "baseline configuration") for record in baseline["fixed"]["images"]
                      if len(record["ground_truth"]) >= 2]
    failures = [case for case in selected_cases if case[0]["fp"] or case[0]["fn"]]
    failures.sort(key=lambda case: (-case[0]["identity_confusions"], -case[0]["conflicting_prediction_pairs"],
                                    -case[0]["fp"] - case[0]["fn"], case[0]["image"]))
    baseline_failures = [case for case in baseline_cases if case[0]["fp"] or case[0]["fn"]]
    baseline_failures.sort(key=lambda case: (-case[0]["fp"] - case[0]["fn"], case[0]["image"]))
    successes = [case for case in selected_cases if case[0]["fp"] == case[0]["fn"] == 0]
    successes.sort(key=lambda case: (-len(case[0]["ground_truth"]), case[0]["image"]))
    chosen, used_images = [], set()
    def add_cases(cases, limit):
        for case in cases:
            if len(chosen) >= limit:
                break
            if case[0]["image"] not in used_images:
                chosen.append(case)
                used_images.add(case[0]["image"])
    add_cases(failures + baseline_failures, count // 2)
    add_cases(successes, count)
    add_cases(failures + baseline_failures + selected_cases, count)
    if len(chosen) < count:
        raise ValueError(f"Only {len(chosen)} unique multi-face test images are available; need {count}.")
    gallery_dir = output_dir / "gallery"
    gallery_dir.mkdir()
    markdown = ["# Detection visualization gallery", "", "Owner: Dario Garza.", "",
                "All images come from the held-out test split. Selection of the model settings used validation only.", "",
                "Ground truth appears on the left. Predictions with identity IDs and confidence scores appear on the right. "
                "Green predictions match the correct identity at IoU >= 0.50; red predictions are unmatched. "
                "A wrong identity contributes both a false positive and a false negative.", "",
                "Failures from the predeclared baseline may supplement failures from the selected configuration. "
                "Each panel names the settings that actually generated its predictions.", ""]
    gallery_rows = []
    for index, (record, configuration, source) in enumerate(chosen, 1):
        filename = f"{index:02d}_{Path(record['image']).stem}_{config_id(configuration)}.jpg"
        status, caption = render_gallery_panel(dataset, record, configuration, gallery_dir / filename, source)
        markdown.extend([f"## {index}. {status} — {record['image']}", "", f"Settings: {source}, `{config_id(configuration)}`.", "",
                         f"![{status}: {record['image']}]({filename})", "", caption, ""])
        gallery_rows.append({"image": record["image"], "panel": filename, "status": status, "source": source,
                             "config_id": config_id(configuration), "caption": caption})
    (gallery_dir / "README.md").write_text("\n".join(markdown), encoding="utf-8")
    write_csv(gallery_dir / "captions.csv", gallery_rows)
    return gallery_rows


def metric_table(rows):
    header = "| Configuration | Split | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | F1 | Mean matched IoU | FP | FN |"
    lines = [header, "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        iou_text = f"{row['mean_matched_iou']:.4f}" if row["mean_matched_iou"] is not None else "N/A"
        lines.append(f"| {row['config_id']} | {row['split']} | {row['map50']:.4f} | {row['map50_95']:.4f} | "
                     f"{row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {iou_text} | {row['fp']} | {row['fn']} |")
    return lines


def write_report(output_dir, summary, validation_rows, test_rows):
    selected = summary["selection"]
    baseline_validation = next(row for row in validation_rows if row["config_id"] == config_id(BASELINE))
    selected_validation = next(row for row in validation_rows if row["config_id"] == selected["config_id"])
    change = selected_validation["f1"] - baseline_validation["f1"]
    lines = ["# Milestone 3 — Hyperparameter sweep", "", "Owner: Dario Garza.", "",
             "## Method", "", "This is an **inference hyperparameter sweep**, using one fixed YOLOv8 checkpoint. "
             "It varies confidence and the IoU threshold used by NMS. Both are explicitly allowed by the milestone. "
             "Class-aware and class-agnostic NMS are compared as an additional control for overlapping identity predictions. "
             "Learning rate, training augmentation and model weights are unchanged.", "",
             f"There are {len(validation_rows)} configurations. Validation contains "
             f"{summary['dataset']['splits']['val']['images']} images and {summary['dataset']['splits']['val']['faces']} faces. "
             "The selection rule is maximum validation micro-F1, then recall, precision, mAP@0.5:0.95, "
             "class-aware NMS and proximity to baseline. This rule is fixed before test evaluation.", "",
             "mAP uses a confidence floor of 0.001 and integrates across confidence cutoffs. "
             "Precision, recall and F1 use each row's stated confidence. Consequently, mAP is shared by rows "
             "with the same NMS settings; changing the reporting threshold alone does not change that AP calculation. "
             "The NMS IoU threshold is separate from the fixed IoU >= 0.50 used to match a prediction to a reference box. "
             "Mean matched IoU includes correct same-identity matches only and excludes misses and false positives. "
             "Matching is one-to-one in descending IoU, consistent with the baseline evaluation script. "
             "Ultralytics' automatic max-F1 precision/recall are saved separately in the run metadata.", "",
             "## Validation comparison", "", *metric_table(validation_rows), "",
             "![Validation F1 sweep](validation_f1_heatmap.png)", "", "## Selected settings and interpretation", "",
             f"Selected: **{selected['config_id']}**. Validation F1 changes from {baseline_validation['f1']:.4f} "
             f"to {selected_validation['f1']:.4f} ({change:+.4f}) against the predeclared baseline. "
             f"Validation false positives change from {baseline_validation['fp']} to {selected_validation['fp']}; "
             f"false negatives change from {baseline_validation['fn']} to {selected_validation['fn']}. "
             "Higher confidence removes weak predictions but can miss faces. Lower NMS IoU suppresses boxes more aggressively. "
             "Class-agnostic NMS can remove different-identity duplicates, but it can also suppress a real nearby face.", "",
             "## Held-out test check", "", *metric_table(test_rows), "",
             "The baseline and selected configuration were the only predeclared test comparisons. "
             "Test results were not used to revise settings. Gallery images were curated after that choice. "
             "These synthetic scenes reuse a small source-photo pool; the results do not establish accuracy on real group photos.", "",
             "## Gallery and handoff", "", "See [the annotated gallery](gallery/README.md) for at least ten unique multi-face examples and captions. "
             "Use `validation_sweep.csv`, `test_metrics.csv`, `per_class_metrics.csv`, `selected_config.json` and this summary "
             "in the team's detection-results document. There are no new training loss curves because this sweep does not retrain the model.", "",
             "## Reproduction", "", f"Checkpoint SHA-256: `{summary['checkpoint_sha256']}`.", "",
             f"Dataset SHA-256: `{summary['dataset']['dataset_sha256']}`.", "",
             "The exact arguments, environment and input provenance are saved in `run_summary.json`. "
             "Low-confidence predictions are retained under `raw_predictions/` so threshold scores can be audited.", "",
             "References: [Ultralytics validation](https://docs.ultralytics.com/modes/val/) and "
             "[Ultralytics prediction/NMS settings](https://docs.ultralytics.com/modes/predict/).", ""]
    (output_dir / "hyperparameter_summary.md").write_text("\n".join(lines), encoding="utf-8")


def run_experiment(checkpoint=None, training_zip=None, dataset=None, output=None, device="cpu", batch=8,
                   image_size=640, confidences=(0.10, 0.25, 0.50, 0.75), nms_ious=(0.30, 0.50, 0.70),
                   agnostic_modes=(False, True), gallery_count=10):
    dataset = Path(dataset or ROOT / "project_milestone02" / "dataset").resolve()
    output = Path(output or ROOT / "project_milestone03" / "results" / "hyperparameter_sweep").resolve()
    grid = build_grid(confidences, nms_ious, agnostic_modes)
    if len(set(confidences)) < 2 or len(set(nms_ious)) < 2:
        raise ValueError("Vary at least two confidence values and two NMS IoU values.")
    if gallery_count < 10:
        raise ValueError("The milestone requires at least ten gallery examples.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output folder is not empty: {output}. Choose a new --output folder.")
    if output == dataset or dataset in output.parents or output in dataset.parents:
        raise ValueError("Output and dataset directories must not overlap.")
    dataset_audit = audit_dataset(dataset)
    with tempfile.TemporaryDirectory(prefix="m3_sweep_") as temporary:
        checkpoint_path = resolve_checkpoint(checkpoint, training_zip, Path(temporary))
        output.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(temporary) / "ultralytics_config"))
        import numpy as np
        import torch
        import ultralytics
        import yaml
        from ultralytics import YOLO
        from ultralytics.utils import SETTINGS
        if ultralytics.__version__ != ULTRALYTICS_VERSION:
            raise RuntimeError(f"Use ultralytics=={ULTRALYTICS_VERSION}; found {ultralytics.__version__}.")
        SETTINGS.update({"sync": False})
        model = YOLO(str(checkpoint_path))
        names = [str(model.names[index]) for index in range(len(model.names))]
        if model.task != "detect" or names != IDENTITIES:
            raise ValueError(f"Expected the five-identity detection checkpoint; found task={model.task}, names={names}.")
        specification = yaml.safe_load((dataset / "data.yaml").read_text(encoding="utf-8"))
        specification["path"] = str(dataset)
        dataset_yaml = Path(temporary) / "data.yaml"
        dataset_yaml.write_text(yaml.safe_dump(specification), encoding="utf-8")
        validation_cache, validation_rows, per_class_rows = {}, [], []
        for configuration in grid:
            key = (configuration["nms_iou"], configuration["agnostic_nms"])
            if key not in validation_cache:
                validation_cache[key] = validate_model(model, dataset_yaml, output, "val", *key, device, batch, image_size)
            prediction_dir, standard = validation_cache[key]
            fixed = evaluate_saved_predictions(dataset, prediction_dir, "val", configuration["confidence"])
            row = {"config_id": config_id(configuration), "split": "val", **configuration,
                   "map50": standard["map50"], "map50_95": standard["map50_95"], **fixed["overall"]}
            validation_rows.append(row)
            for identity, scores in fixed["per_class"].items():
                per_class_rows.append({"split": "val", "config_id": row["config_id"], "identity": identity, **scores,
                                       "map50": standard["per_class"].get(identity, {}).get("map50", 0.0),
                                       "map50_95": standard["per_class"].get(identity, {}).get("map50_95", 0.0)})
            print(f"{row['config_id']}: validation F1={row['f1']:.4f}, FP={row['fp']}, FN={row['fn']}", flush=True)
        write_csv(output / "validation_sweep.csv", validation_rows)
        selection = choose_configuration(validation_rows)
        chosen_configuration = {key: selection[key] for key in BASELINE}
        # Persist the validation-only decision before opening the held-out split.
        write_json(output / "selected_config.json", {"selection_split": "val", "rule": "maximum micro-F1, then recall, precision, AP, class-aware mode, baseline proximity",
                                                    "config_id": selection["config_id"], **chosen_configuration,
                                                    "image_size": image_size, "validation_metrics": selection})
        test_cache, test_results, test_rows = {}, {}, []
        for label, configuration in [("baseline", BASELINE), ("selected", chosen_configuration)]:
            key = (configuration["nms_iou"], configuration["agnostic_nms"])
            if key not in test_cache:
                test_cache[key] = validate_model(model, dataset_yaml, output, "test", *key, device, batch, image_size)
            prediction_dir, standard = test_cache[key]
            fixed = evaluate_saved_predictions(dataset, prediction_dir, "test", configuration["confidence"])
            test_results[label] = {"configuration": configuration, "standard": standard, "fixed": fixed}
            test_rows.append({"config_id": config_id(configuration), "split": "test", **configuration,
                              "map50": standard["map50"], "map50_95": standard["map50_95"], **fixed["overall"]})
            for identity, scores in fixed["per_class"].items():
                per_class_rows.append({"split": "test", "config_id": config_id(configuration), "identity": identity, **scores,
                                       "map50": standard["per_class"].get(identity, {}).get("map50", 0.0),
                                       "map50_95": standard["per_class"].get(identity, {}).get("map50_95", 0.0)})
        write_csv(output / "test_metrics.csv", test_rows)
        write_csv(output / "per_class_metrics.csv", per_class_rows)
        write_json(output / "test_predictions.json", test_results)
        gallery = create_gallery(dataset, output, test_results["selected"], test_results["baseline"], gallery_count)
        plot_sweep(validation_rows, output / "validation_f1_heatmap.png")
        checkpoint_hash = sha256(checkpoint_path)
        reference_path = ROOT / "project_milestone03" / "results" / "metrics.json"
        reference = json.loads(reference_path.read_text(encoding="utf-8")) if reference_path.is_file() else {}
        summary = {"created_utc": datetime.now(timezone.utc).isoformat(), "owner": "Dario Garza", "experiment": "inference_threshold_sweep",
                   "checkpoint_sha256": checkpoint_hash,
                   "matches_aditi_reference_checkpoint": checkpoint_hash == reference.get("checkpoint_sha256"),
                   "training_zip_sha256": sha256(training_zip) if training_zip else None,
                   "dataset": dataset_audit, "selection": selection,
                   "environment": {"python": platform.python_version(), "ultralytics": ultralytics.__version__,
                                   "torch": torch.__version__, "numpy": np.__version__, "device": device},
                   "evaluation": {"image_size": image_size, "batch": batch, "workers": 0,
                                  "ap_confidence_floor": AP_CONFIDENCE_FLOOR, "match_iou": MATCH_IOU,
                                  "matching": "same class, one-to-one, descending IoU", "rect": True, "augment": False,
                                  "confidence_values": list(confidences), "nms_iou_values": list(nms_ious),
                                  "agnostic_nms_values": list(agnostic_modes)},
                   "validation": validation_rows, "test": test_rows,
                   "standard_metrics": {"val": [{"nms_iou": key[0], "agnostic_nms": key[1], **value[1]}
                                               for key, value in validation_cache.items()],
                                        "test": [{"nms_iou": key[0], "agnostic_nms": key[1], **value[1]}
                                                 for key, value in test_cache.items()]},
                   "gallery": gallery}
        write_json(output / "run_summary.json", summary)
        write_report(output, summary, validation_rows, test_rows)
        print(f"Finished. Selected on validation: {selection['config_id']}. Outputs: {output}", flush=True)
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--checkpoint", type=Path, help="Path to Aditi's trained best.pt.")
    inputs.add_argument("--training-zip", type=Path, help="Training results ZIP containing best.pt.")
    parser.add_argument("--dataset", type=Path, default=ROOT / "project_milestone02" / "dataset")
    parser.add_argument("--output", type=Path, default=ROOT / "project_milestone03" / "results" / "hyperparameter_sweep")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--confidences", nargs="+", type=float, default=[0.10, 0.25, 0.50, 0.75])
    parser.add_argument("--nms-ious", nargs="+", type=float, default=[0.30, 0.50, 0.70])
    parser.add_argument("--class-aware-only", action="store_true")
    parser.add_argument("--gallery-count", type=int, default=10)
    args = parser.parse_args()
    run_experiment(checkpoint=args.checkpoint, training_zip=args.training_zip, dataset=args.dataset, output=args.output,
                   device=args.device, batch=args.batch, image_size=args.image_size, confidences=args.confidences,
                   nms_ious=args.nms_ious, agnostic_modes=(False,) if args.class_aware_only else (False, True),
                   gallery_count=args.gallery_count)


if __name__ == "__main__":
    main()
