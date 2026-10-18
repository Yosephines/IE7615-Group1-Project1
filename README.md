# IE 7615 - Group 1 - Project 1

We are building a system that recognizes celebrities and locates their faces in group images.

## Start here

- **Milestone 4 (final):** Read the [final report](output%20and%20pdf/milestone04_final_report.pdf), then follow the [Milestone 4 guide](project_milestone04/README.md). One command detects and names every face in an image: the [YOLOv8n detector](project_milestone03/models/best.pt) finds faces and the [ResNet18 classifier](project/models/group1_repro_v1/resnet18_frozen_backbone.pt) votes on who they are. Try the [live demo](project_milestone04/demo.py), browse the [results gallery](project_milestone04/gallery/README.md) or read the [test results](project_milestone04/results/summary.md).
- **Milestone 3:** Read the [detection results](output%20and%20pdf/milestone03_detection_results.pdf), then follow the [Milestone 3 guide](project_milestone03/README.md). It covers the [live demo](project_milestone03/demo.py), the [trained YOLOv8n checkpoint](project_milestone03/models/best.pt), the [hyperparameter sweep](project_milestone03/results/dario_sweep_20261010_010347_535395/hyperparameter_summary.md) and the [annotated gallery](project_milestone03/results/dario_sweep_20261010_010347_535395/gallery/README.md).
- **Milestone 2:** Open the [dataset notebook](project_milestone02/milestone02_dataset_pipeline.ipynb). It shows how we build group images, label faces and split the data. The examples and results are saved in the notebook. See the [Milestone 2 guide](project_milestone02/README.md) to run it.
- **Milestone 1:** Read the [proposal and training results](output%20and%20pdf/proposal_and_training_results.pdf). Follow [notebooks 01-05](notebooks/README.md) for the classification work, or the [setup guide](project/SETUP.md) to run it.

## Quick start

Python 3.12 or 3.13, CPU only. From the repo folder (Windows: see the [Milestone 4 guide](project_milestone04/README.md#setup)):

```sh
python3.13 -m venv .venv-m3
source .venv-m3/bin/activate
python -m pip install torch==2.14.0 torchvision==0.29.0
python -m pip install -r project_milestone04/requirements.txt

python project_milestone04/demo.py                               # live demo at http://127.0.0.1:7860
python project_milestone04/pipeline.py path/to/photo.jpg folder/  # any number of images, saved to tmp/m4_pipeline/
```

## Reproduce the reported numbers

Every trained model is committed, so nothing needs retraining. Each command runs in about a minute or less on a laptop CPU.

| Milestone | Command | Compare with |
| --- | --- | --- |
| 4 - pipeline | `python project_milestone04/evaluate.py --output tmp/m4_reproduce` | [`project_milestone04/results/`](project_milestone04/results/) (identical on macOS ARM) |
| 4 - gallery | `python project_milestone04/gallery.py` | [`project_milestone04/gallery/`](project_milestone04/gallery/) |
| 3 - detector | `python project_milestone03/hyperparameter_sweep.py --checkpoint project_milestone03/models/best.pt --output tmp/m3_reproduce` | [sweep run](project_milestone03/results/dario_sweep_20261010_010347_535395/) |
| 2 - dataset | Run the [dataset notebook](project_milestone02/milestone02_dataset_pipeline.ipynb) ([guide](project_milestone02/README.md)); it rebuilds `project_milestone02/dataset/` | [`summary.json`](project_milestone02/dataset/summary.json) |
| 1 - classifiers | `python project/scripts/verify_saved_run.py` ([guide](project/SETUP.md); also runs in the environment above) | [`project/results/group1_repro_v1/`](project/results/group1_repro_v1/) |

Regression checks: `python -m unittest discover -s project_milestone04 -p "test_*.py"` and the same for `project_milestone03`. Retraining instructions are in the [Milestone 1 setup guide](project/SETUP.md) and the [Milestone 3 guide](project_milestone03/README.md#test-results-selected-settings).

## Folders

| Folder | Contents |
| --- | --- |
| `project_milestone04/` | End-to-end pipeline, live demo, evaluation, results and the final gallery |
| `project_milestone03/` | YOLOv8 checkpoint, Milestone 3 demo, detection evaluation, hyperparameter sweep and gallery |
| `project_milestone02/` | Main detection-data notebook, finished dataset and supporting code |
| `notebooks/` | Milestone 1 classification notebooks |
| `output and pdf/` | Milestone 1 proposal and training results, Milestone 3 detection results, Milestone 4 final report |
| `data/` | Original celebrity photos used by all milestones |
| `project/` | Milestone 1 code, the trained classifiers and results, plus archived team work |

## Milestone 4 results

Held-out test: 36 synthetic scenes (113 faces) and 17 single-face portraits. Settings were chosen on validation. "Unknown" means the face was boxed but its name withheld because the two models were not confident enough.

| Method | Scenes: correct / wrong / unknown | Scene precision | Portraits: correct / wrong / unknown |
| --- | --- | ---: | --- |
| YOLOv8n alone (Milestone 3) | 104 / 8 / 0 | 0.920 | 0 / 1 / 0 (16 missed) |
| ResNet18 alone (Milestone 1) | 71 / 41 / 0 | 0.634 | 11 / 6 / 0 |
| **Pipeline** | **85 / 1 / 26** | **0.988** | **16 / 0 / 1** |

The pipeline finds faces in close-up portraits that the detector alone missed. It also withholds most wrong names, at the cost of withholding some correct ones. See the [Milestone 4 guide](project_milestone04/README.md#results-held-out-test) for the full table and limitations.

## Milestone 3 results

| Metric (held-out test, 36 scenes, 113 faces) | Score |
| --- | --- |
| mAP@0.5 | 0.962 |
| mAP@0.5:0.95 | 0.862 |
| Precision | 0.929 |
| Recall | 0.920 |
| Mean IoU of correct matches | 0.932 |

YOLOv8n fine-tuned for 50 epochs. Confidence 0.50 and NMS IoU 0.70 were selected on validation. Run the demo or reproduce these numbers:

```sh
python project_milestone03/demo.py
python project_milestone03/hyperparameter_sweep.py --checkpoint project_milestone03/models/best.pt --output tmp/m3_reproduce
```

See the [Milestone 3 guide](project_milestone03/README.md) for setup and per-identity results.

## Milestone 1 results

| Model | Correct test photos | Accuracy |
| --- | --- | --- |
| Small CNN | 5 of 15 | 33.3% |
| Deeper CNN | 6 of 15 | 40.0% |
| ResNet18 | 10 of 15 | 66.7% |

We selected ResNet18 using separate validation photos before checking the test results. With only 15 test photos, these scores are an initial comparison.

## Celebrity subset

We selected five identities from the shared class collection, with enough photos to use the same number for each person. The table credits the classmates who claimed these identities in the class sheet.

| Identity ID | Available photos | Claimed by | Group |
| --- | --- | --- | --- |
| 7007 | 24 | Mus Ab Irfan Yilmaz | 3 |
| 2970 | 25 | Rhea Paul | 3 |
| 2336 | 25 | Masato Kan | 3 |
| 7 | 24 | Jin-woo Hong | 2 |
| 4428 | 23 | David Fung | 4 |

Milestone 1 uses 23 photos per person: 17 for training, 3 for validation and 3 for testing. Milestone 2 keeps those assignments and adds the six remaining photos to validation and test.
