# Project roadmap

| Milestone | Due | Main work |
| --- | --- | --- |
| 1 â€” Classification baseline | End of Module 3 | Compare three classifiers, evaluate on held-out images, and save the best model |
| 2 â€” Synthetic dataset | End of Module 4 | Create multi-celebrity composite images and YOLO-format bounding-box annotations |
| 3 â€” Detection | Mid-Module 5 | Fine-tune YOLOv8 on the synthetic dataset and evaluate detection metrics |
| 4 â€” Integrated system | End of Module 6 | Combine the components, complete the final report, and verify repository quality and reproducibility |

## Keep the classification model
The recommended Milestone 1 classifier is frozen-backbone ResNet18 with a five-class output layer. Preserve its selected checkpoint, class order and preprocessing. It becomes the classification component of the later integrated system. YOLOv8 is a separate detector; the ResNet checkpoint is not a YOLOv8 checkpoint.

See [the model handoff](best_model_handoff.md) for the required files.

## Organization as the project grows
Keep the current baseline under `group1_repro_v1`. The earlier `aditi_5id` results remain historical. Never mix results between runs.

For each later milestone, add only the folders that have actual content:
- notebooks/milestone2/, notebooks/milestone3/, notebooks/milestone4/ for milestone-specific workflows.
- configs/ for dataset/model settings and versioned split manifests.
- src/ for shared implementation.
- logs/<run>/ and results/<run>/ for execution records, metrics and figures.
- docs/milestone2/, docs/milestone3/, docs/milestone4/ for methods and reports.
- data/ and models/ for dataset/model artifacts. The small Milestone 1 subset and checkpoints are included; keep future large artifacts in documented storage with checksums.

Record the dataset version, seed, class mapping, dependency versions, checkpoint location and command/notebook order for every run. Add a release tag or commit hash for each submission. Verify that all reported figures can be traced to saved logs or predictions.
