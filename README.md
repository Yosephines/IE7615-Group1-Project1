# IE 7615 · Group 1 · Project 1

**Team:** Yosephine Tong, Dario Garza and Aditi

We are building a system that identifies celebrities in photos. For Milestone 1, we trained three models to recognize five people from the CelebA dataset.

## Celebrity subset

We selected five identities from the shared class collection, with enough photos to use the same number for each person. The table credits the classmates who claimed these identities in the class sheet.

| Identity ID | Available photos | Claimed by | Group |
| --- | --- | --- | --- |
| 7007 | 24 | Mus Ab Irfan Yilmaz | 3 |
| 2970 | 25 | Rhea Paul | 3 |
| 2336 | 25 | Masato Kan | 3 |
| 7 | 24 | Jin-woo Hong | 2 |
| 4428 | 23 | David Fung | 4 |

We use 23 photos per person: 17 for training, 3 for validation and 3 for testing. Six remaining photos are unused.

## Results

| Model | Correct test predictions | Accuracy |
| --- | --- | --- |
| Small CNN | 5 of 15 | 33.3% |
| Deeper CNN | 6 of 15 | 40.0% |
| ResNet18 | 10 of 15 | 66.7% |

We chose **ResNet18** because it also did best on the separate validation photos used to select a model. It builds on earlier training with a large image dataset. The two custom CNNs learn from our photos from the start.

Our test set is small, so these results are a starting point. We need more photos to judge how well the model works in different conditions.

## Read the reports

- [Project proposal — 1 page](output/pdf/group1_proposal.pdf)
- [Training results — 3 pages](output/pdf/group1_training_results.pdf) · [Read as Markdown](docs/training_results.md)
- [Dataset details](docs/dataset.md) · [Full comparison table](results/model_comparison.csv)

## Open the notebooks in order

All five contain completed results.

1. [Prepare the data](notebooks/01_data_preparation.ipynb) — divide photos into training, validation and test groups.
2. [Train the small CNN](notebooks/02_small_custom_cnn.ipynb).
3. [Train ResNet18](notebooks/03_resnet18_frozen_backbone.ipynb).
4. [Train the deeper CNN](notebooks/04_deeper_custom_cnn.ipynb).
5. [Compare the models](notebooks/05_evaluation_and_comparison.ipynb) — choose using validation, then measure test accuracy.

## Check or repeat the work

The photos, saved models, training logs and test predictions are included.

Follow the [setup and run instructions](docs/reproducing_results.md). They explain how to check the saved models without training, or train all three again while keeping the submitted results.

The current experiment is named `group1_repro_v1`. Files are organized as follows:

| Folder | What it contains |
| --- | --- |
| `data/identities/` | 121 photos; 115 are used after balancing the five people |
| `configs/` | Photo lists, training settings and software versions |
| `notebooks/`, `src/`, `scripts/` | Notebooks, shared code and commands to run the project |
| `models/`, `logs/`, `results/` | Trained models, training records, scores and charts |
| `docs/`, `output/pdf/` | Written explanations and submission reports |

Dario's and Aditi's original notebooks are preserved. Aditi's earlier results are kept under `aditi_5id`, separate from the results above.
