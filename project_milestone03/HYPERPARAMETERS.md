# Dario's section — Hyperparameter sweep and detection gallery

Open [milestone03_hyperparameters_and_gallery.ipynb](milestone03_hyperparameters_and_gallery.ipynb) in Google Colab. Run the cells in order. The notebook can load an uploaded project ZIP and the trained checkpoint, so it does not require Google Drive or GitHub access.

## Required input

The project contains the full Milestone 2 dataset, but the supplied project ZIP **does not contain Aditi's YOLOv8 checkpoint**. The three `.pt` files under `project/models/` are Milestone 1 classification models and cannot be used for detection.

Provide either:

- Aditi's `best.pt`, or
- `milestone03_training_results.zip`, containing one `best.pt` at any directory depth.

The existing evaluation identifies Aditi's checkpoint by SHA-256:

```text
64f5686b5a7760fee18cf215b1e29f4aa5af95dc90fbf0b571635ed08aa1d2e6
```

The new run records whether the supplied checkpoint matches that reference. It also requires the same five identity IDs and class order. It does not silently substitute a pretrained COCO model or a classification checkpoint.

## What the sweep does

| Parameter | Values |
|---|---|
| Confidence threshold | 0.10, 0.25, 0.50, 0.75 |
| IoU threshold for NMS | 0.30, 0.50, 0.70 |
| NMS class mode | Class-aware and class-agnostic |

This gives **24 inference configurations** of one fixed model. Confidence and NMS IoU are two of the hyperparameters permitted by the assignment. Class-agnostic NMS is an extra comparison motivated by the baseline's overlapping predictions with different identity IDs. The script does not compare new training runs or change learning rates.

All configurations are scored on validation. The winner maximizes micro-F1; ties use recall, precision, mAP@0.5:0.95, class-aware NMS and proximity to baseline, in that order. The decision is saved before evaluating the held-out test split. Test evaluates only the selected settings and the predeclared baseline: confidence 0.25, NMS IoU 0.70, class-aware NMS.

For comparable mAP, validation always uses a low confidence floor of 0.001. Fixed-threshold precision, recall, F1 and mean matched IoU are then calculated from the saved boxes at each configured confidence. mAP is therefore repeated across confidence rows sharing the same NMS settings. The script keeps the distinction visible in its tables and generated interpretation.

Boxes are matched one-to-one to the correct identity at IoU >= 0.50, using descending IoU to remain consistent with `evaluate.py`. That matching threshold is independent of the NMS IoU threshold.

## Run locally

Use the Milestone 3 environment described in [README.md](README.md). Alternatively, create a separate environment using Python 3.12 or 3.13 and install:

```sh
python -m pip install -r project_milestone03/requirements_hyperparameters.txt
```

From the repository root, use one input route:

```sh
python project_milestone03/hyperparameter_sweep.py --checkpoint /path/to/best.pt
```

```sh
python project_milestone03/hyperparameter_sweep.py --training-zip /path/to/milestone03_training_results.zip
```

On Colab, the notebook automatically selects a GPU when available. Locally, the default is CPU; use `--device 0` for the first CUDA GPU. `--batch 8` and `--image-size 640` reproduce the baseline's evaluation size. `--class-aware-only` produces the smaller 12-configuration comparison.

Outputs default to `project_milestone03/results/hyperparameter_sweep/`. A non-empty output folder is rejected to prevent stale prediction files from contaminating another run. Choose another `--output` folder when repeating an experiment. The notebook automatically gives every run a new timestamped folder.

## Outputs and handoff

| Output | Purpose |
|---|---|
| `validation_sweep.csv` | All 24 validation configurations, mAP, fixed-threshold metrics and error counts |
| `validation_f1_heatmap.png` | Confidence/NMS trade-offs for each class mode |
| `selected_config.json` | Settings selected using validation only |
| `test_metrics.csv` | Baseline and selected settings on held-out test |
| `per_class_metrics.csv` | Per-identity AP, precision, recall, F1 and matched IoU |
| `hyperparameter_summary.md` | Method, tables and numerical interpretation for Yosephine's report |
| `gallery/README.md` | Ten distinct multi-face examples with captions |
| `gallery/*.jpg` | Ground truth alongside labelled predictions and confidence scores |
| `gallery/captions.csv` | Captions and the exact configuration for each image |
| `test_predictions.json` | Auditable test boxes and one-to-one matches |
| `run_summary.json` | Dataset/checkpoint hashes, environment, arguments and all results |
| `raw_predictions/` | Low-confidence boxes retained for score auditing |

The gallery prioritizes actual failures and clear successes. If the selected settings have too few failures, failures from the baseline can be included; each image states which configuration produced it. If no failure exists under either tested configuration, none is invented. The gallery is descriptive and must not be used to retune on test.

The sweep does not generate additional training loss curves because it does not retrain the model. Keep Aditi's existing training log and Yosephine's existing baseline report. The generated summary and CSVs supply the missing comparison section for that report.

## Verification

```sh
python project_milestone03/test_hyperparameter_sweep.py
```

These checks cover box overlap, duplicate predictions, wrong identities, confidence filtering, missing detections, invalid labels, source-photo leakage, baseline data hashes and validation-only selection. See [verification.md](verification.md) for the status of execution in this workspace.

Official parameter references: [validation](https://docs.ultralytics.com/modes/val/) and [prediction/NMS](https://docs.ultralytics.com/modes/predict/).
