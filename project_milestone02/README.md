# Milestone 2 - Face detection dataset

**Start with the [main notebook](milestone02_dataset_pipeline.ipynb).** It follows the work from the original photos to the finished dataset, with labelled examples shown inline.

The [dataset](dataset/) contains **576 images and 1,820 labelled faces**, about **33 MB**. It is included in the repo. Milestone 3 will use [data.yaml](dataset/data.yaml) to load it into YOLOv8.

## Run the notebook

Use Python 3.13. From the main repository folder, create a separate environment:

```sh
python -m venv .venv-m2
```

Activate it with `.venv-m2\Scripts\Activate.ps1` on Windows PowerShell, or `source .venv-m2/bin/activate` on macOS/Linux. Then run:

```sh
python -m pip install -r project_milestone02/requirements.txt
python -m ipykernel install --sys-prefix --name ie7615-m2 --display-name "Milestone 2"
python -m jupyterlab
```

Open `project_milestone02/milestone02_dataset_pipeline.ipynb`, choose the **Milestone 2** kernel, and select **Run > Run All Cells**. Running all cells replaces `dataset/`. To read the saved results, just open the notebook without running it.

The first run downloads the small YuNet face detector and checks its file hash. Seed 47 fixes the random choices. Small pixel differences can occur between computers.

## Files

| File or folder | Purpose |
| --- | --- |
| `milestone02_dataset_pipeline.ipynb` | Main walkthrough, checks and image examples |
| `m2_dataset.py` | Functions called by the notebook; no separate run needed |
| `requirements.txt` | Software needed to run the notebook |
| `dataset/` | Images and labels in `train/`, `val/` and `test/` |

Within `dataset/`, each split has `images/` and `labels/`. `manifest.csv` records each face's source photo and box. `source_split.csv` records the photo assignments; `summary.json` records the counts.

## How we built it

We use the same five identities as Milestone 1. Each 800 x 600 image combines 2-5 different people on a drawn room background. Layout, face size and lighting vary; photo proportions are preserved.

YuNet finds the face boxes. The source-photo record supplies each identity. Frames are rebuilt if an expected face is not found or the detector finds an extra face without a label.

Each image has a YOLO text file with one row per face:

```text
class_id x_center y_center width height
```

Coordinates are fractions of the image width and height. Classes 0-4 represent IDs **7007, 2970, 2336, 7 and 4428**, in that order.

## Split

Source photos are split first, so the same photo cannot appear in training and test scenes. We keep the Milestone 1 assignments and add its six unused photos to validation and test.

| | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| Source photos | 85 | 19 | 17 |
| Original compositions | 168 | 36 | 36 |
| Augmented copies | 336 | 0 | 0 |
| Final images | 504 | 36 | 36 |
| Labelled faces | 1,596 | 111 | 113 |

The **70/15/15 split** applies to the original compositions. Each training image gets two augmented copies. Every identity appears in every split.

## Augmentation

Training images and boxes are transformed together. Validation and test receive no extra copies. `p` is the chance of applying a transform.

| Transform | Settings | Purpose |
| --- | --- | --- |
| Horizontal flip | p=0.5 | Left/right variation |
| Scale, shift and rotation | Scale 0.85-1.15; shift +/-5%; rotate +/-7 degrees; p=0.8 | Different framing |
| Brightness and contrast | +/-0.2 each; p=0.7 | Lighting variation |
| Hue, saturation and value | +/-5, +/-15, +/-10; p=0.4 | Colour variation |
| Gamma | 85-115; p=0.3 | Exposure variation |
| Gaussian or motion blur | Kernel 3-5 or 3-7; p=0.25 | Focus and motion |
| Noise | Standard deviation 0.01-0.04; p=0.25 | Camera noise |
| JPEG compression | Quality 60-95; p=0.3 | Compression effects |

Copies are retried if boxes lose too much area, become too narrow, an extra face is detected, or the image barely changes. We avoid vertical flips, large rotations and erasing faces.

## Checks and limits

The notebook checks file pairs, box bounds, class coverage, duplicate files and shared source photos. It also compares boxes with fresh YuNet detections and shows examples for visual review. Using the same detector checks consistency; it does not prove every annotation is correct.

These are synthetic collages. Validation and test reuse only 3-4 source photos per identity, so they do not measure performance on a wide range of real group photos.

## Team

- **Aditi:** initial image composition.
- **Dario:** YuNet annotation method and prototype labels.
- **Murat:** combined pipeline, revised composition, augmentation and source-photo split.
- **Yosephine:** notebook review, final organization and documentation.

The [earlier prototype](../project/archive/milestone02_prototype/) is archived separately.
