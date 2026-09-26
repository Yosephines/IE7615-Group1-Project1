# IE 7615 · Group 1 · Project 1

Celebrity identification and detection with CelebA.

## Current work

[Dario's training notebook](notebooks/milestone01_team01_dario.ipynb) contains data preparation and completed training/validation runs for two custom CNNs and a pretrained ResNet18 classifier. It is the current implementation for Milestone 1.

The notebook uses **PyTorch / torchvision in Google Colab**, five identity IDs, and a shared balanced split. The introduction has been corrected to identity **4428** to match the code and saved outputs.

**Held-out test evaluation is pending.** The following numbers are saved validation results at the checkpoint with the lowest validation loss, not test results:

| Architecture | Selected epoch | Validation accuracy | Total parameters | Reported training time |
| --- | ---: | ---: | ---: | ---: |
| Small CNN: 32, 64 channels | 29 | 33.33% | 19,717 | 18.9 s |
| Deeper CNN: 32, 64, 128, 256 channels | 28 | 53.33% | 389,701 | 21.7 s |
| ResNet18: frozen backbone, new classifier | 30 | 80.00% | 11,179,077 | 21.9 s |

ResNet18 is the leading validation candidate. Final test metrics and the written carry-forward decision remain to be completed. See [training results](docs/training_results.md) for interpretation and limitations.

## Data

Selected IDs, in classifier order: **7007, 2970, 2336, 7, 4428**.

The saved run uses 23 images per identity: 17 training, 3 validation, and 3 test. Total: **85 training / 15 validation / 15 test images**. Test images are reserved but not evaluated in this notebook.

See [dataset documentation](docs/dataset.md) for counts, transforms, and split details. Raw images and checkpoints are not stored in Git.

## Repository contents

| Location | Contents |
| --- | --- |
| [notebooks/](notebooks/) | Dario's combined notebook and its handoff guide |
| [configs/baseline.json](configs/baseline.json) | Settings transcribed from the notebook; a reference record, not an executable configuration |
| [docs/dataset.md](docs/dataset.md) | Recorded identities, image counts, preprocessing, and missing split artifact |
| [docs/training_results.md](docs/training_results.md) | Validation results and remaining evaluation work |
| [docs/proposal.md](docs/proposal.md) | Updated proposal draft with unconfirmed team assignments marked |
| [results/model_comparison.csv](results/model_comparison.csv) | Actual validation summary, with test and inference fields left empty |
| [logs/](logs/) | Instructions for adding the existing training-history CSVs |
| [data/](data/) | Local dataset placement guide |
| [models/](models/) | Local checkpoint and team handoff guide |

## Open and run

1. Clone this private repository after accepting your collaborator invitation:
   ```sh
   git clone https://github.com/Yosephines/IE7615-Group1-Project1.git
   cd IE7615-Group1-Project1
   ```
2. Open `notebooks/milestone01_team01_dario.ipynb` in Google Colab. To view saved results, no training is necessary.
3. To rerun, select a GPU runtime and set `DATA_DIR` to your Drive directory containing the five identity folders. Set `OUTPUT_DIR` to a checkpoint/output directory you control.
4. Run the notebook from top to bottom. It mounts Google Drive and trains all three architectures for 30 epochs each. Reusing the same output directory overwrites the corresponding checkpoint/history filenames.
5. Copy the three history CSVs into `logs/`. Keep checkpoints in shared team storage and document their locations.

The code imports torch, torchvision, pandas, Pillow, matplotlib, IPython, and google.colab. Exact package versions were not recorded in the supplied run, so a tested dependency lock file is still needed. The current notebook is Colab-specific.

**Before evaluating existing checkpoints:** obtain the exact split manifest and checkpoints from Dario. The notebook builds the split in memory but does not export the `split.csv` mentioned in its narrative. Regenerating with the same seed only reproduces it if the original file inventory is unchanged. See the [handoff guide](notebooks/README.md).

## Remaining Milestone 1 work

- [ ] Obtain and commit the exact split manifest and three training-history CSVs.
- [ ] Share the three selected checkpoints outside Git and record their checksums/locations.
- [ ] Record tested dependency versions and hardware details.
- [ ] Evaluate each selected checkpoint on the same held-out test split.
- [ ] Add per-class accuracy or a confusion matrix and training-loss curves.
- [ ] Complete the best-model justification using measured trade-offs.
- [ ] Organize final deliverables into a data-preparation notebook and one training notebook per architecture, as required by the assignment.
- [ ] Finalize team responsibilities and export the one-page proposal to PDF.
- [ ] Finalize the 2–4-page training-results document.
- [ ] Verify all teammates and teaching staff have accepted collaborator invitations.
- [ ] Submit the private repository link, submission commit/tag, and documents through one team member.

## Contributions

Dario Garza (`TheMorrisGitHub`) contributed the combined training/validation notebook. Repository owner: [Yosephines](https://github.com/Yosephines).

Use branches and pull requests for team changes. Commit code, small logs, configuration records, and figures; keep raw datasets, credentials, and checkpoint binaries out of Git.
