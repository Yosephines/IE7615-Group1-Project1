# Verification status — Dario's Milestone 3 section

Verified on October 9, 2026.

## Completed checks

- **12 regression checks passed.** They cover IoU geometry, one-to-one matching, duplicate detections, wrong identities, confidence filtering, missing predictions, invalid labels, zero-match IoU, the 24-configuration grid, validation-only selection, checkpoint ZIP handling and dataset provenance.
- **The supplied dataset matches the existing baseline evaluation exactly.** The canonical image/label hash is `2b31b489a30f70d1a0f24f75e87641ee137774c87f6c335362c7e6142a99b782`. Its 504 training images, 36 validation images and 36 test images contain 1,596, 111 and 113 labelled faces. Source photos do not cross split boundaries, and the scene manifest uses sources from the correct split.
- **The notebook schema and every code cell compile.** All nine code cells were then executed in order with a Python runner and a real Ultralytics YOLOv8 architecture on the supplied dataset. The smoke run completed 24 validation configurations, two predeclared test comparisons, selection from validation only, CSV/JSON/Markdown export, a heatmap, ten distinct annotated gallery images and result ZIP creation.
- **The non-empty detection path also passed.** A deliberately modified untrained model produced real validator detections. Saved-confidence filtering and per-class AP extraction completed successfully.
- **Gallery rendering was visually reviewed** using a fixture with a correct identity, a wrong identity and a missing face. Both ground-truth and prediction boxes, confidence scores and captions were legible.

The smoke execution took approximately 24 seconds on this CPU environment. That time is not a runtime estimate for the trained checkpoint or Colab.

## Environment used

| Component | Version |
|---|---|
| Python | 3.12.14 |
| Ultralytics | 8.4.174 |
| PyTorch | 2.14.0+cpu |
| Torchvision | 0.29.0+cpu |
| NumPy | 2.3.5 |
| Device | CPU |

The notebook preserves Colab's existing PyTorch/CUDA installation and records the environment of each actual run. Interactive Colab uploads and the Colab download UI were not exercised in this local verification.

## Trained-model execution

The sweep was then run on Aditi's trained checkpoint, now committed as [`models/best.pt`](models/best.pt) (SHA-256 `64f5686b5a7760fee18cf215b1e29f4aa5af95dc90fbf0b571635ed08aa1d2e6`, matching the baseline reference). The results are in [`results/dario_sweep_20261010_010347_535395/`](results/dario_sweep_20261010_010347_535395/). The untrained smoke outputs and the rendering fixture above are software checks, not submission results.

On October 9, 2026 the run was repeated from the committed checkpoint on macOS ARM CPU (Python 3.13.15, Ultralytics 8.4.174, PyTorch 2.14.0) in 24 seconds:

```sh
python project_milestone03/hyperparameter_sweep.py --checkpoint project_milestone03/models/best.pt --output tmp/m3_reproduce
```

Validation selected the same settings (`conf_0.50_iou_0.70_aware`). On test, the selected settings reproduced the committed true/false positive and miss counts, precision, recall and F1 exactly; mean matched IoU differed by less than 0.0001. mAP@0.5 was 0.9616 instead of 0.9623 and mAP@0.5:0.95 was 0.8615 instead of 0.8623; the baseline had 11 false positives instead of 12. These are floating-point differences between platforms and do not change any conclusion.
