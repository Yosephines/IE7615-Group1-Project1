# IE 7615 - Group 1 - Project 1

We trained three models to recognize five celebrities from photos. ResNet18 performed best, identifying 10 of 15 test photos correctly.

## Start here

1. **Read the [Proposal and training results](output%20and%20pdf/proposal_and_training_results.pdf).** It brings the proposal, results and charts into one document.
2. **Open [notebooks 01-05](notebooks/README.md)** to follow the work from data preparation to evaluation. They already include results.
3. **To run the code**, follow the [setup guide](project/SETUP.md). You do not need to run each script separately.

## What is in each folder?

| Folder | What to look for |
| --- | --- |
| **output and pdf/** | The combined report, plus separate proposal and results PDFs |
| **notebooks/** | The five numbered steps of the project |
| **data/** | The photos used for training and testing |
| **project/** | Supporting code, saved models, logs and original team notebooks |

## Results

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

We use 23 photos per person: 17 for training, 3 for validation and 3 for testing. Six remaining photos are unused.
