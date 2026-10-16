"""Milestone 4: build the final results gallery from the pipeline's saved predictions.

Owner: Murat Akca (draft for Dario to curate). Reads results/predictions.csv and results/ta_check.csv written by
evaluate.py, re-runs the pipeline on the chosen images to draw them, and writes gallery/:

- single_correct/        20 correctly identified single-face portraits (test first, then validation)
- single_misclassified/  every single-face error: portraits named wrongly or withheld, and TA-grid strangers
                         that still received a celebrity name, each with a short automatic analysis
- multi_face/            10 held-out test scenes with the pipeline's boxes and names
- *_sheet.jpg            one contact sheet per section, for the report
- README.md              captions and analysis

    python project_milestone04/gallery.py
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import pipeline as m4
from evaluate import DATASET, TA_IMAGE, TA_KNOWN_TILES
from pipeline import IDS, UNKNOWN

ROOT = m4.ROOT
RESULTS = ROOT / "project_milestone04" / "results"
GALLERY = ROOT / "project_milestone04" / "gallery"
CORRECT_COUNT, MULTI_COUNT = 20, 10


def font(size):
    for name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def with_caption(rgb, lines, width=None):
    """Image above a white strip of caption lines."""
    image = Image.fromarray(rgb)
    if width and image.width != width:
        image = image.resize((width, round(image.height * width / image.width)), Image.LANCZOS)
    text_font = font(max(12, image.width // 32))
    line_height = text_font.size + 6
    canvas = Image.new("RGB", (image.width, image.height + line_height * len(lines) + 8), "white")
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    for index, line in enumerate(lines):
        draw.text((6, image.height + 4 + index * line_height), line, fill=(32, 50, 68), font=text_font)
    return canvas


def contact_sheet(paths, target, columns):
    images = [Image.open(path) for path in paths]
    cell_w, cell_h = max(i.width for i in images), max(i.height for i in images)
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_w + (columns + 1) * 8, rows * cell_h + (rows + 1) * 8), (226, 230, 234))
    for index, image in enumerate(images):
        sheet.paste(image, (8 + (index % columns) * (cell_w + 8), 8 + (index // columns) * (cell_h + 8)))
    sheet.save(target, quality=90)


def opinions(face):
    detector = (f"YOLO {face['detector_identity']} {float(face['detector_score']):.2f}"
                if face["detector_identity"] else "YOLO no box")
    return (f"{detector}; ResNet18 {face['classifier_identity']} {float(face['classifier_probability']):.2f}; "
            f"fused {float(face['confidence']):.2f}")


def analysis(truth, face, threshold):
    """One-sentence explanation of a single-face error from the two models' opinions."""
    detector, classifier = face["detector_identity"], face["classifier_identity"]
    named = face["predicted_identity"] if "predicted_identity" in face else face["identity"]
    overlap = face.get("iou")
    if overlap not in (None, "") and float(overlap) < 0.5:
        return (f"Localization failure: the detector's box covers little of the face (IoU {float(overlap):.2f} with "
                f"the aligned face position), so both models judged the wrong region and the answer is unreliable.")
    if named == UNKNOWN:
        fused = float(face["confidence"])
        if detector == classifier == truth:
            return (f"Withheld: both models chose {truth}, but not confidently enough; the fused confidence "
                    f"{fused:.3f} is just below the {threshold:.2f} threshold.")
        right = [name for name, who in (("YOLO", detector), ("ResNet18", classifier)) if who == truth]
        if right:
            return (f"Withheld: only {right[0]} chose {truth}; the disagreement lowered the fused confidence to "
                    f"{fused:.3f}, below the {threshold:.2f} threshold.")
        return f"Withheld: neither model chose {truth}; rejecting the low-confidence answer avoided a wrong name."
    if truth not in IDS:
        if detector == classifier:
            return (f"Stranger named {named}: both models agreed, so the fused confidence stayed above the "
                    f"threshold. A closed-set model has no way to say 'none of the five'.")
        return (f"Stranger named {named}: YOLO's confident score ({float(face['detector_score']):.2f}) carried "
                f"the fused confidence over the threshold although ResNet18 disagreed.")
    if detector == classifier == named:
        return f"Both models confused {truth} with {named}; agreement made the fused confidence high."
    if face["stage"] == "whole_image":
        return "No face box was found, so ResNet18 classified the whole photo on its own."
    return (f"The models disagreed (YOLO {detector or '-'}, ResNet18 {classifier}); the averaged scores still "
            f"favoured {named} above the threshold.")


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def build(output=GALLERY):
    pipeline = m4.CelebrityPipeline()
    threshold = pipeline.settings["unknown_threshold"]
    rows = read_csv(RESULTS / "predictions.csv")
    sections = {name: output / name for name in ("single_correct", "single_misclassified", "multi_face")}
    for folder in sections.values():
        folder.mkdir(parents=True, exist_ok=True)
        for old in folder.glob("*.jpg"):
            old.unlink()
    readme = {name: [] for name in sections}

    # Single-face portraits: test first (never used for any choice), then validation.
    portraits = [row for row in rows if row["kind"] == "portrait"]
    portraits.sort(key=lambda row: (row["split"] != "test", row["image"]))
    correct = [row for row in portraits if row["outcome"] == "correct"][:CORRECT_COUNT]
    errors = [row for row in portraits if row["outcome"] != "correct"]
    for index, row in enumerate(correct + errors, 1):
        path = ROOT / row["image"]
        faces, _ = pipeline(path)
        rgb = np.asarray(Image.fromarray(m4.load_rgb(path)).resize((356, 436), Image.LANCZOS))
        scaled = [{**face, "box": None if face["box"] is None else [v * 2 for v in face["box"]]} for face in faces]
        is_error = row["outcome"] != "correct"
        lines = [f"{row['split']} | true {row['true_identity']} -> {row['predicted_identity']}",
                 opinions(row)]
        name = f"{index:02d}_{row['split']}_{row['true_identity']}_{Path(row['image']).stem}.jpg"
        section = "single_misclassified" if is_error else "single_correct"
        with_caption(m4.draw(rgb, scaled), lines).save(sections[section] / name, quality=90)
        entry = {"file": name, "split": row["split"], "truth": row["true_identity"],
                 "prediction": row["predicted_identity"], "opinions": opinions(row)}
        if is_error:
            entry["analysis"] = analysis(row["true_identity"], row, threshold)
        readme[section].append(entry)

    # TA real-photo grid: strangers that still received a name are misclassifications too.
    ta_rgb = m4.load_rgb(TA_IMAGE)
    ta_faces, analysis_result = pipeline(TA_IMAGE)
    height, width = analysis_result["size"]
    for face in sorted(ta_faces, key=lambda face: (round(face["box"][1] * 3 // height), face["box"][0])):
        cx, cy = (face["box"][0] + face["box"][2]) / 2, (face["box"][1] + face["box"][3]) / 2
        tile = (int(cy * 3 // height), int(cx * 3 // width))
        truth = TA_KNOWN_TILES.get(tile, "stranger")
        if face["identity"] == truth or (truth == "stranger" and face["identity"] == UNKNOWN):
            continue
        top, left = tile[0] * height // 3, tile[1] * width // 3
        crop = ta_rgb[top:top + height // 3, left:left + width // 3]
        shifted = {**face, "box": [face["box"][0] - left, face["box"][1] - top, face["box"][2] - left,
                                   face["box"][3] - top]}
        crop = np.asarray(Image.fromarray(m4.draw(crop, [shifted])).resize((356, 356), Image.LANCZOS))
        row = {**face, "predicted_identity": face["identity"]}
        name = f"ta_tile_r{tile[0]}c{tile[1]}.jpg"
        lines = [f"TA grid tile ({tile[0]}, {tile[1]}) | true: {truth} -> {face['identity']}", opinions(row)]
        with_caption(crop, lines).save(sections["single_misclassified"] / name, quality=90)
        readme["single_misclassified"].append({"file": name, "split": "TA image", "truth": truth,
                                               "prediction": face["identity"], "opinions": opinions(row),
                                               "analysis": analysis(truth, row, threshold)})

    # Multi-face scenes: 10 test scenes, the most crowded first, half of them with at least one non-correct face.
    by_scene = defaultdict(list)
    for row in rows:
        if row["kind"] == "scene" and row["split"] == "test":
            by_scene[row["image"]].append(row)
    clean = [s for s, r in by_scene.items() if all(x["outcome"] == "correct" for x in r)]
    flawed = [s for s, r in by_scene.items() if s not in clean]
    crowd = lambda scene: (-len(by_scene[scene]), scene)  # noqa: E731
    chosen = sorted(clean, key=crowd)[:MULTI_COUNT // 2] + sorted(flawed, key=crowd)[:MULTI_COUNT // 2]
    for scene in sorted(chosen):
        path = DATASET / scene
        faces, _ = pipeline(path)
        counts = defaultdict(int)
        for row in by_scene[scene]:
            counts[row["outcome"]] += 1
        truth = ", ".join(sorted((row["true_identity"] for row in by_scene[scene] if row["true_identity"]),
                                 key=IDS.index))
        summary = ", ".join(f"{counts[key]} {key}" for key in ("correct", "unknown", "wrong", "missed",
                                                                "extra_named") if counts[key])
        name = f"{Path(scene).stem}.jpg"
        with_caption(m4.draw(path, faces), [f"{Path(scene).stem} | faces: {truth}", summary]).save(
            sections["multi_face"] / name, quality=90)
        notes = [f"{row['true_identity']} withheld (YOLO {row['detector_identity']}, ResNet18 "
                 f"{row['classifier_identity']}, fused {float(row['confidence']):.2f})"
                 for row in by_scene[scene] if row["outcome"] == "unknown"]
        readme["multi_face"].append({"file": name, "truth": truth, "summary": summary, "notes": "; ".join(notes)})

    for section, columns in (("single_correct", 5), ("single_misclassified", 4), ("multi_face", 2)):
        contact_sheet(sorted(sections[section].glob("*.jpg")), output / f"{section}_sheet.jpg", columns)
    write_readme(output, readme, threshold)
    return readme


def write_readme(output, readme, threshold):
    correct, errors, multi = readme["single_correct"], readme["single_misclassified"], readme["multi_face"]
    lines = [
        "# Milestone 4 results gallery",
        "",
        "Built by [`gallery.py`](../gallery.py) from the end-to-end pipeline's saved predictions "
        "([`results/predictions.csv`](../results/predictions.csv)). Each label shows the pipeline's answer and "
        "fused confidence; captions also give each model's own answer. Grey boxes are faces the pipeline found "
        f"but did not name, because the fused confidence was below the validated threshold of {threshold:.2f}.",
        "",
        "Portraits are the held-out source photos (178 × 218) shown at twice their size. Test photos come first; "
        "validation photos were used only to choose the pipeline settings, never for training.",
        "",
        f"## {len(correct)} correctly identified single faces",
        "",
        "![Correct single faces](single_correct_sheet.jpg)",
        "",
        "| File | Split | Identity | Model opinions |",
        "| --- | --- | --- | --- |",
    ]
    lines += [f"| [{e['file']}](single_correct/{e['file']}) | {e['split']} | {e['truth']} | {e['opinions']} |"
              for e in correct]
    lines += ["", f"## {len(errors)} single-face errors with analysis", "",
              "Portraits the pipeline named wrongly or withheld, and faces of people outside our five identities "
              "(TA real-photo grid) that still received a celebrity name.", "",
              "![Single-face errors](single_misclassified_sheet.jpg)", "",
              "| File | Source | Truth | Pipeline | Model opinions | Analysis |", "| --- | --- | --- | --- | --- | --- |"]
    lines += [f"| [{e['file']}](single_misclassified/{e['file']}) | {e['split']} | {e['truth']} | "
              f"{e['prediction']} | {e['opinions']} | {e['analysis']} |" for e in errors]
    lines += ["", f"## {len(multi)} multi-face test scenes", "",
              "Held-out test scenes (800 × 600). Five have every face named correctly; five include at least one "
              "face that was withheld or missed.", "", "![Multi-face scenes](multi_face_sheet.jpg)", "",
              "| File | Faces (reference) | Result | Notes |", "| --- | --- | --- | --- |"]
    lines += [f"| [{e['file']}](multi_face/{e['file']}) | {e['truth']} | {e['summary']} | {e['notes']} |"
              for e in multi]
    lines += ["", "Rebuild: `python project_milestone04/evaluate.py` then `python project_milestone04/gallery.py`.", ""]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=GALLERY)
    args = parser.parse_args()
    readme = build(args.output)
    print({section: len(entries) for section, entries in readme.items()}, "->", args.output)


if __name__ == "__main__":
    main()
