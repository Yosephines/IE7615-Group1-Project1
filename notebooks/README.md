# Start with notebook 01

These five notebooks are the main workflow. Each includes completed output, so you can read it without rerunning training.

| Order | Notebook | What it does |
| --- | --- | --- |
| 01 | [Prepare data](01_data_preparation.ipynb) | Divides photos into training, validation and test groups |
| 02 | [Small CNN](02_small_custom_cnn.ipynb) | Trains the small custom model |
| 03 | [ResNet18](03_resnet18_frozen_backbone.ipynb) | Trains the final layer of the pretrained model |
| 04 | [Deeper CNN](04_deeper_custom_cnn.ipynb) | Trains the larger custom model |
| 05 | [Compare results](05_evaluation_and_comparison.ipynb) | Selects using validation, then checks test predictions |

To run them, use the [setup guide](../project/SETUP.md). Choose a new experiment name when retraining so the saved results are preserved.

These numbered notebooks were created by separating Dario's data preparation and training work and Aditi's evaluation work into the required steps. Their saved outputs come from the new Group 1 training run.

Shared code is in `project/src/`. Dario's and Aditi's original combined notebooks are preserved unchanged in `project/archive/`.
