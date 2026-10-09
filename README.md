# IE 7615 - Group 1 - Project 1

We are building a system that recognizes celebrities and locates their faces in group images.

## Start here

- **Milestone 3 — Dario:** Open the [hyperparameter and gallery notebook](project_milestone03/milestone03_hyperparameters_and_gallery.ipynb). See the [sweep guide](project_milestone03/HYPERPARAMETERS.md) for the required trained checkpoint and execution status.
- **Milestone 2:** Open the [dataset notebook](project_milestone02/milestone02_dataset_pipeline.ipynb). It shows how we build group images, label faces and split the data. The examples and results are saved in the notebook. See the [Milestone 2 guide](project_milestone02/README.md) to run it.
- **Milestone 1:** Read the [proposal and training results](output%20and%20pdf/proposal_and_training_results.pdf). Follow [notebooks 01-05](notebooks/README.md) for the classification work, or the [setup guide](project/SETUP.md) to run it.

## Folders

| Folder | Contents |
| --- | --- |
| `project_milestone02/` | Main detection-data notebook, finished dataset and supporting code |
| `project_milestone03/` | Baseline detection evaluation, hyperparameter sweep and visualization gallery code |
| `notebooks/` | Milestone 1 classification notebooks |
| `output and pdf/` | Milestone 1 proposal and training results |
| `data/` | Original celebrity photos used by both milestones |
| `project/` | Milestone 1 code and results, plus archived team work |

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
