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

## Pending trained-model execution

The supplied project ZIP does not include Aditi's `best.pt` or `milestone03_training_results.zip`. Only metrics, the training log and training settings are present under `project_milestone03/results/`. Milestone 1 `.pt` files are classification checkpoints and are unsuitable for this detection sweep.

**No sweep scores from Aditi's trained model or final detection gallery are claimed yet.** The untrained smoke outputs and the rendering fixture are excluded from the deliverable project archive. They are software checks, not submission results.

Provide the checkpoint or training-results ZIP and run the notebook to generate the final numerical comparison and annotated gallery. The script records checkpoint SHA-256 and whether it matches the baseline reference:

```text
64f5686b5a7760fee18cf215b1e29f4aa5af95dc90fbf0b571635ed08aa1d2e6
```

If a different trained checkpoint is supplied, the baseline and selected settings are evaluated again for that checkpoint; historical scores are not reused as results of the new run.
