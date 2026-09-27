# Milestone 1 notebooks

The numbered notebooks are the submission workflow. All were executed on the included dataset for `group1_repro_v1`.

| Order | Notebook | Purpose |
| --- | --- | --- |
| 01 | `01_data_preparation.ipynb` | Check images and save the exact balanced split |
| 02 | `02_small_custom_cnn.ipynb` | Train the two-block CNN |
| 03 | `03_resnet18_frozen_backbone.ipynb` | Train the pretrained ResNet18 output layer |
| 04 | `04_deeper_custom_cnn.ipynb` | Train the four-block CNN |
| 05 | `05_evaluation_and_comparison.ipynb` | Select using validation, then evaluate held-out test images |

Architecture definitions, transforms and training are shared in [src/pipeline.py](../src/pipeline.py). This meets the separate data-preparation and per-architecture notebook requirement without duplicating implementation.

Use the [README commands](../README.md#reproduce) to run a new experiment. They preserve these submitted outputs. For interactive use, select the project Python environment and set a new `GROUP1_RUN_ID` in the kernel before notebook 01; use that same ID in every notebook. The default ID is the submitted run, whose checkpoints cannot be overwritten.

## Original contributions

- `milestone01_team01_dario.ipynb`: Dario's original combined preparation/training notebook.
- `milestone01_team01 - outputtest.ipynb`: Aditi's original combined training/evaluation notebook.

These originals retain their historical code and outputs. Their Colab paths and older results are not the current reproduction instructions. Aditi's results are stored separately as `aditi_5id`.
