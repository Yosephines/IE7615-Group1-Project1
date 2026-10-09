# Milestone 3 - Evaluation and report

Start with the [results report](../output%20and%20pdf/milestone03_detection_results.pdf). It covers Aditi's model, the test scores, loss curves and TA image check.

Dario's inference sweep and detection gallery are available in the
[hyperparameter notebook](milestone03_hyperparameters_and_gallery.ipynb).
See [the sweep guide](HYPERPARAMETERS.md) for inputs, methods, outputs and verification.
It compares confidence and NMS thresholds using one fixed checkpoint, with validation-only
selection. Numerical sweep results still require Aditi's trained `best.pt`, which is not
included in the supplied project ZIP.

## Files

| File or folder | Purpose |
| --- | --- |
| `evaluate.py` | Runs the evaluation and builds the report |
| `hyperparameter_sweep.py` | Runs Dario's threshold comparison and annotated test gallery |
| `milestone03_hyperparameters_and_gallery.ipynb` | Colab walkthrough for Dario's section |
| `HYPERPARAMETERS.md` | Sweep method, required checkpoint and handoff instructions |
| `test_hyperparameter_sweep.py` | Metric and dataset regression checks |
| `results/` | Metrics, original training log and training settings |

## Run again

Use Python 3.13. From the main repo folder:

```powershell
python -m venv .venv-m3
.venv-m3\Scripts\Activate.ps1
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu torch==2.14.0+cpu torchvision==0.29.0+cpu
python -m pip install ultralytics==8.4.174 numpy==2.5.2 pandas==3.0.6 `
  matplotlib==3.11.2 pillow==12.3.0 opencv-python==5.0.0.93 `
  reportlab==5.0.1 pyyaml==6.0.3
```

Set the path to the folder containing Aditi's files and the TA image:

```powershell
$inputFolder = "C:\path\to\Milestone3"
python project_milestone03/evaluate.py `
  --training-zip "$inputFolder\milestone03_training_results.zip" `
  --ta-image "$inputFolder\testimage.jpeg" `
  --training-notebook "$inputFolder\milestone03_yolov8_training (1).ipynb"
```

This evaluates the saved model without retraining. It replaces the evaluation results and Milestone 3 report. The original input files stay unchanged.

Extra prediction files and report images are generated in `tmp/m3_details/` and stay out of GitHub.
