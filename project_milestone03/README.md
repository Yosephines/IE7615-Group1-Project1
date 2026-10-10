# Milestone 3 - YOLOv8 transfer learning and evaluation

We fine-tuned a pretrained YOLOv8n on the Milestone 2 synthetic group scenes to detect and identify five celebrities. Start with the [results report](../output%20and%20pdf/milestone03_detection_results.pdf).

| Deliverable | Where |
| --- | --- |
| Trained checkpoint | [`models/best.pt`](models/best.pt) (YOLOv8n, 6 MB) |
| Training notebook, log and settings | [`training/`](training/milestone03_yolov8_training.ipynb), [`results/training_history.csv`](results/training_history.csv), [`results/training_args.yaml`](results/training_args.yaml) |
| Loss curves and validation plots | [`results/training_plots/`](results/training_plots/) |
| Detection-results document | [`milestone03_detection_results.pdf`](../output%20and%20pdf/milestone03_detection_results.pdf) |
| Hyperparameter sweep | [`hyperparameter_summary.md`](results/dario_sweep_20261010_010347_535395/hyperparameter_summary.md), method in [`HYPERPARAMETERS.md`](HYPERPARAMETERS.md) |
| Visualization gallery | [`gallery/README.md`](results/dario_sweep_20261010_010347_535395/gallery/README.md) (10 captioned test scenes) |
| Live demo | [`demo.py`](demo.py) |
| TA image check (unknown faces, real photos) | [`results/ta_image_check/`](results/ta_image_check/README.md) |

## Setup

Use Python 3.12 or 3.13. From the main repo folder:

macOS / Linux:

```sh
python3.13 -m venv .venv-m3
source .venv-m3/bin/activate
python -m pip install torch==2.14.0 torchvision==0.29.0
python -m pip install -r project_milestone03/requirements_demo.txt
```

Windows (PowerShell):

```powershell
python -m venv .venv-m3
.venv-m3\Scripts\Activate.ps1
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu torch==2.14.0+cpu torchvision==0.29.0+cpu
python -m pip install -r project_milestone03/requirements_demo.txt
```

A GPU is not needed. Everything below runs on CPU in under a minute.

## Live demo

```sh
python project_milestone03/demo.py
```

Open <http://127.0.0.1:7860>. Upload a photo, take a webcam snapshot or click one of the held-out test scenes. The app draws each box with its identity ID and confidence score and lists the boxes in a table. The sliders change the confidence and NMS IoU thresholds live; they start at the settings selected on validation (0.50 and 0.70). Add `--share` for a temporary public link, for example on Colab.

Without a browser, run it on one image:

```sh
python project_milestone03/demo.py --image project_milestone02/dataset/test/images/test_0005.jpg
```

This prints each detection and saves the annotated copy to `tmp/demo/test_0005_detections.jpg` (or `--output`). `test_0005` is a clean success; `test_0002` shows the model's most common failure, identity 7007 labelled as 7.

The model only knows five identities. A face of anyone else is either missed or given one of the five IDs, so a confident box is not proof of identity. The [TA image check](results/ta_image_check/README.md) shows this on real photos: all nine faces are boxed, but the six unknown people all receive a celebrity ID, some with confidence up to 0.97.

## Reproduce the reported metrics

One command evaluates the committed checkpoint. It runs the 24-setting validation sweep, chooses settings on validation, scores the held-out test split and rebuilds the per-class table and gallery:

```sh
python project_milestone03/hyperparameter_sweep.py --checkpoint project_milestone03/models/best.pt --output tmp/m3_reproduce
```

Compare `tmp/m3_reproduce/test_metrics.csv` and `per_class_metrics.csv` with the committed [`dario_sweep_20261010_010347_535395`](results/dario_sweep_20261010_010347_535395/) run. On macOS ARM CPU we got the same selected settings and identical counts, precision, recall and F1 for them. mAP@0.5 differed in the fourth decimal (0.9616 vs 0.9623) and mean matched IoU by less than 0.0001, because floating-point results vary slightly by platform.

`evaluate.py` rebuilds the PDF report. It also needs Aditi's training ZIP, the training notebook and the TA image, which are not in the repo:

```sh
python project_milestone03/evaluate.py --training-zip milestone03_training_results.zip --ta-image testimage.jpeg \
  --training-notebook project_milestone03/training/milestone03_yolov8_training.ipynb \
  --sweep-summary project_milestone03/results/dario_sweep_20261010_010347_535395/run_summary.json
```

## Final model and hyperparameters

Training (Aditi Rao, Colab Tesla T4, 8.2 minutes). Full settings are in [`training_args.yaml`](results/training_args.yaml).

| Setting | Value |
| --- | --- |
| Starting checkpoint | `yolov8n.pt` (COCO-pretrained) |
| Data | Milestone 2 `data.yaml`: 504 train / 36 val / 36 test scenes, 5 classes |
| Epochs | 50; best validation mAP@0.5:0.95 at epoch 29, saved as `best.pt` |
| Image size, batch, seed | 640, 8, 42 (deterministic) |
| Optimizer | `auto`, which chose AdamW with learning rate 0.001111, momentum 0.9 |
| Augmentation | Ultralytics defaults: mosaic 1.0 (off for last 10 epochs), horizontal flip 0.5, HSV 0.015/0.7/0.4, translate 0.1, scale 0.5, RandAugment, erasing 0.4 |
| Loss weights | box 7.5, cls 0.5, dfl 1.5 |

Inference, selected on validation by Dario's sweep:

| Setting | Baseline | Selected |
| --- | --- | --- |
| Confidence threshold | 0.25 | **0.50** |
| NMS IoU threshold | 0.70 | **0.70** |
| NMS mode | class-aware | **class-aware** |

Checkpoint SHA-256: `64f5686b5a7760fee18cf215b1e29f4aa5af95dc90fbf0b571635ed08aa1d2e6`.

## Test results (selected settings)

36 held-out scenes, 113 faces. A correct match needs the right identity and IoU >= 0.50.

| Identity | AP@0.5 | AP@0.5:0.95 | Precision | Recall | Mean matched IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| 7007 | 0.931 | 0.816 | 1.000 | 0.625 | 0.921 |
| 2970 | 0.995 | 0.891 | 1.000 | 1.000 | 0.937 |
| 2336 | 0.995 | 0.899 | 1.000 | 1.000 | 0.933 |
| 7 | 0.900 | 0.803 | 0.821 | 1.000 | 0.932 |
| 4428 | 0.990 | 0.902 | 0.864 | 1.000 | 0.936 |
| **All** | **0.962** | **0.862** | **0.929** | **0.920** | **0.932** |

Nearly every error is an identity confusion. Of the 9 faces of 7007 that were missed, 5 were labelled 7, 3 were labelled 4428 and 1 got no box. These 8 confusions are all 8 false positives, which explains both 7007's low recall and the lower precision for 7 and 4428.

Retraining from scratch: open [`training/milestone03_yolov8_training.ipynb`](training/milestone03_yolov8_training.ipynb) in Colab, or run `yolo detect train data=project_milestone02/dataset/data.yaml model=yolov8n.pt epochs=50 imgsz=640 batch=8 seed=42`. GPU nondeterminism means a retrained checkpoint will not match the committed one byte for byte.

## Files

| File or folder | Purpose |
| --- | --- |
| `demo.py` | Gradio app and command-line demo |
| `models/best.pt` | Final fine-tuned YOLOv8n checkpoint |
| `training/` | Aditi's Colab training notebook |
| `evaluate.py` | Runs the baseline evaluation and builds the report |
| `hyperparameter_sweep.py` | Runs Dario's threshold comparison and annotated test gallery |
| `milestone03_hyperparameters_and_gallery.ipynb` | Colab walkthrough for the sweep and gallery |
| `HYPERPARAMETERS.md` | Sweep method and outputs |
| `test_hyperparameter_sweep.py` | Metric and dataset regression checks |
| `results/` | Metrics, training log, training plots and the sweep run |
