# IE 7615 — Group 1 — Project 1

Celebrity identification and detection using a discriminative computer-vision pipeline.

## Milestone 1: Classification Baseline

**Status:** Repository scaffold only. Dataset selection, training, and evaluation are pending.
**Team:** Group 1

**Repository owner:** [Yosephines](https://github.com/Yosephines)

**Framework:** To be confirmed by the team.

### Planned comparisons
- Custom CNN trained from scratch (required).
- ImageNet-pretrained model, such as ResNet (required).
- Additional architecture or fine-tuning variant (recommended).

## Repository layout

| Path | Purpose |
| --- | --- |
| `notebooks/01_data_preparation.ipynb` | Identity selection, preprocessing, and saved per-identity splits |
| `notebooks/02_custom_cnn.ipynb` | Custom CNN training and evaluation |
| `notebooks/03_transfer_learning.ipynb` | Pretrained baseline training and evaluation |
| `notebooks/04_architecture_variant.ipynb` | Optional third comparison |
| `src/` | Shared data, model, training, and evaluation code |
| `configs/` | Versioned experiment settings |
| `data/` | Local-only dataset; see its README |
| `logs/` | Small CSV training logs and environment records |
| `results/` | Model-comparison table and exported figures |
| `models/` | Local-only checkpoints |
| `docs/` | Proposal and training-results templates |

## Dataset and identities

Use 4–6 distinct CelebA identity IDs. Confirm names independently before associating them with IDs; do not infer a person's name from appearance. Record selection rationale, counts, and the class-to-index mapping in `docs/dataset.md`.

Save one reproducible, per-identity train/validation/test split and reuse it for all architectures. Ensure no image appears in multiple splits. Apply random augmentation only to training data; keep validation/test transforms deterministic. Choose models and hyperparameters using validation results, then evaluate the final choices on the held-out test set.

## Getting started

1. Clone the private repository:
   ```sh
   git clone https://github.com/Yosephines/IE7615-Group1-Project1.git
   cd IE7615-Group1-Project1
   ```
2. Choose the framework and record exact tested dependencies in `requirements.txt` or an environment file.
3. Obtain CelebA through its authorized source and place files locally as described in `data/README.md`.
4. Complete and run the data-preparation notebook, then the training notebooks in order.
5. Save each run's configuration, seed, dependency versions, hardware, epoch metrics, training duration, and checkpoint location.
6. Update `results/model_comparison.csv` and `docs/training_results.md` with actual measured results.

Notebook files are planning templates, not runnable training implementations yet. Add exact installation and execution commands here when the implementation is ready.

## Team workflow

- Use a branch for each change and open a pull request into `main`.
- Keep raw images, credentials, and model checkpoints out of Git.
- Commit small training logs, loss/accuracy plots, comparison tables, and completed notebook artifacts.
- Review notebook outputs for embedded dataset images and private local information before committing.
- Record task owners and future milestone responsibilities in `docs/proposal.md`.

## Milestone 1 submission checklist

- [ ] Keep the repository private.
- [ ] Invite all teammates and teaching staff with read/write access; verify accepted invitations.
- [ ] Document 4–6 identities, counts, selection rationale, and preprocessing.
- [ ] Complete data-preparation notebook and one training notebook per architecture.
- [ ] Compare at least the custom CNN and transfer-learning baseline.
- [ ] Report held-out test accuracy, parameter count, and training time.
- [ ] Include per-class accuracy or a confusion matrix and training-loss curves.
- [ ] Justify the model selected for Milestones 2 and 3.
- [ ] Export the refreshed one-page proposal to PDF.
- [ ] Complete the 2–4-page training-results document (PDF or Markdown export).
- [ ] Record the submitted commit with `git rev-parse HEAD` or create a release tag.
- [ ] Have one team member submit the repository link, commit/tag, and documents on Canvas.
