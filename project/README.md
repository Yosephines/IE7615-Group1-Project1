# Supporting files

Follow the [setup guide](SETUP.md) to run the project. The numbered notebooks are the main workflow; you do not need to open each helper script.

| Folder | Contents |
| --- | --- |
| `src/` | Shared model, training and evaluation code used by the notebooks |
| `scripts/` | Tools for running notebooks, checking models and building reports |
| `configs/` | Dataset inventories, split files and software versions |
| `models/` | Saved trained models |
| `logs/` | Training loss, accuracy and settings |
| `results/` | Evaluation tables, predictions and charts |
| `archive/` | Original team notebooks and earlier training records |

`group1_repro_v1` is the experiment used in the report. That name connects its models, logs and results. The chosen model is `models/group1_repro_v1/resnet18_frozen_backbone.pt`.
