# Milestone 2 · Synthetic multi-celebrity detection dataset

**Location:** [`project_milestone02/dataset/`](dataset/) · **Size:** 576 images (504 train / 36 val / 36 test), 1,820 labelled faces, 33 MB · **Format:** YOLO (one `.txt` per image) with [`data.yaml`](dataset/data.yaml) for YOLOv8

**Notebook:** [`milestone02_dataset_pipeline.ipynb`](milestone02_dataset_pipeline.ipynb) builds the whole dataset, from the raw CelebA photos in `data/identities/` to the finished train/val/test folders. It includes inline annotated samples. The code is in [`m2_dataset.py`](m2_dataset.py).

## Rebuild

```sh
pip install -r project_milestone02/requirements.txt
python project_milestone02/m2_dataset.py
```

To run the notebook instead, also install Jupyter (`pip install jupyter`), then run `jupyter nbconvert --to notebook --execute --inplace project_milestone02/milestone02_dataset_pipeline.ipynb`.

All randomness is seeded. Rebuilding with the script from a fresh clone in a new environment produced byte-identical files (1,156 files compared). Building takes about 30 seconds on a laptop CPU, and the YuNet face-detector model (230 KB) is downloaded once and checked against a pinned SHA-256. The script replaces `dataset/` completely.

## Contents

```
dataset/
├── data.yaml          # train/val/test paths (relative), nc: 5, names 0-4 = 7007, 2970, 2336, 7, 4428
├── train/images, train/labels    # 168 composed frames + 336 augmented copies
├── val/images,   val/labels      # 36 composed frames
├── test/images,  test/labels     # 36 composed frames
├── manifest.csv       # one row per labelled face: split, image, parent frame, identity, source CelebA photo, box, seed
├── source_split.csv   # the split of every source photo
└── summary.json       # counts and check results
```

Label rows: `class_id x_center y_center width height`, normalized to the 800 × 600 image. The class ids match the Milestone 1 classifier (`project/src/pipeline.py`).

## Split

| | Train | Val | Test |
| --- | ---: | ---: | ---: |
| Composed frames (70 / 15 / 15) | 168 | 36 | 36 |
| Augmented copies | 336 | 0 | 0 |
| Images | 504 | 36 | 36 |
| Faces | 1,596 | 111 | 113 |
| Faces of 7007 / 2970 / 2336 / 7 / 4428 | 321 / 348 / 318 / 306 / 303 | 21 / 26 / 20 / 24 / 20 | 24 / 23 / 24 / 23 / 19 |
| Source photos used | 85 | 19 | 17 |

**Rationale.** A source photo is reused across many frames. Splitting finished frames at random would place the same face photo in both train and test. We therefore split the 121 **source photos** first and compose each split only from its own photos:
- The Milestone 1 split is reused: 17 / 3 / 3 photos per identity.
- The 6 photos Milestone 1 left out are alternated between val and test.

The checks in `validate()` confirm:
- no source photo is shared between splits;
- no two image files are identical;
- every identity appears in every split.

Proportions refer to composed frames; augmentation adds copies to train only, so val and test stay untouched originals. Because of this split, Milestone 3 test faces are also unseen by the Milestone 1 classifier.

## Composition

Each frame shows 2–5 **different** identities (69 / 93 / 51 / 27 frames with 2 / 3 / 4 / 5 faces), arranged as a group photo. This extends the team's earlier approach of pasting crops on a shared background:

- **Background:** a procedural room with a wall gradient, floor, and randomly placed whiteboards, windows, posters and doors, under soft directional light. This varies scenes without third-party photos.
- **Layout:** one row, or two rows with the back row 78–90 % of the front-row height, standing in the gaps. A per-frame camera-distance factor of 0.62–1.0 applies. Labelled faces are 55–160 px wide (median 90).
- **Blending:** each head gets a soft elliptical mask plus a neck, shoulder and clothing silhouette. A desk is drawn in front of the group in about 40 % of frames. The CelebA aspect ratio is kept.
- **Lighting:** per-person brightness of 0.85–1.15 and a ±4 % colour tint, plus per-frame exposure, colour cast and slight blur.
- **Validity:**
  - A layout is rejected if a face would leave the frame, be narrower than 36 px, or be more than 3 % covered by another person.
  - 61 train, 11 val and 14 test layouts were rejected and recomposed.

## Annotation

The team's YuNet method is kept: the OpenCV Zoo face detector, run on each frame exactly as it is saved.
- **Labels:** Each placed person must match exactly one detection with confidence ≥ 0.70 and IoU ≥ 0.5 against the face position expected from its source photo. The detected box is the label.
- **Bystander rule:** Any other face-like detection (confidence ≥ 0.50) that overlaps no label rejects the frame. This removes bystanders at the edges of source photos. One train frame was recomposed for this reason.
- **Independent check (notebook section 6):** YuNet was re-run on every saved augmented image. The transformed labels match fresh detections with median IoU 0.923 (5th percentile 0.857, minimum 0.767). No face is missed, and no unlabelled face-like detection remains.

## Augmentation (train only, 2 copies per frame)

| Transform | Parameters | Why |
| --- | --- | --- |
| Horizontal flip | p = 0.5 | Faces are left-right symmetric; doubles pose variety |
| Affine | scale 0.85–1.15, translate ±5 %, rotate ±7°, p = 0.8 | Camera distance, framing, slight tilt; small angles and ellipse-based box rotation keep boxes tight |
| Brightness / contrast | ±0.2 / ±0.2, p = 0.7 | Indoor lighting and exposure |
| Hue / saturation / value | ±5 / ±15 / ±10, p = 0.4 | White balance and camera colour, kept small for plausible skin tones |
| Gamma | 85–115, p = 0.3 | Non-linear exposure |
| Gaussian or motion blur | kernel 3–5 / 3–7, p = 0.25 | Defocus and hand shake |
| Gaussian noise | std 0.01–0.04, p = 0.25 | Sensor noise in dim rooms |
| JPEG compression | quality 60–95, p = 0.3 | Phone and messaging compression |

**Not used:**
- vertical flip and large rotations, which are unrealistic and loosen boxes;
- cutout, which could hide a labelled face;
- mosaic and mixup, which YOLOv8 already applies online in Milestone 3.

**Implementation:**
- Boxes are transformed with the images (albumentations 2.0.8, YOLO format).
- A copy is resampled if a face would lose more than 5 % of its box outside the frame, shrink below 29 px, show an unlabelled face, or be practically identical to its original or the other copy. There were 60 resamples in total.
- Each transform is seeded separately. A single seeded `Compose` in albumentations 2.0.8 draws the same random number for every transform, which made the transforms all-or-nothing.

## Milestone 3 hand-off

Ultralytics 8.4 reads `data.yaml` directly (504 / 36 / 36 images, 0 corrupt, 5 classes), and a 1-epoch YOLOv8n smoke run trained and evaluated without errors:

```sh
yolo detect train data=project_milestone02/dataset/data.yaml model=yolov8n.pt imgsz=640
```

YOLOv8's own online augmentation also runs during training. Consider lowering its `degrees`/`scale` settings, or comparing runs with and without the `_aug` files.

**Limitations:**
- Val and test frames reuse only 3–4 photos per identity, in new layouts. This is the cost of a leak-free split.
- Backgrounds are procedural, and heads come from tightly cropped CelebA photos. Frames are plausible group photos, not photorealistic ones.

## Earlier files in this folder

`synthetic_images/`, `yolo_labels/`, `synthetic_placements.csv`, `face_annotations.csv`, `classes.txt`, `class_mapping.csv` and `milestone02_data_preparation.ipynb` are the team's first prototype:
- 50 frames, flat background;
- `synthetic_images/` contains 45 images while `yolo_labels/` contains 50 label files;
- source photos were not recorded, so the frames could not be split without leakage.

They are kept for reference. Their composition and YuNet annotation ideas are carried into the pipeline above.
