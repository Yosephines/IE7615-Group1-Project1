# Milestone 1 · Training and validation results

**Status: partial results document.** Based on saved output in [Dario's notebook](../notebooks/milestone01_team01_dario.ipynb). Training has not been independently rerun. This document must be completed with test evaluation and figures before submission.

## Setup

Five identities: 7007, 2970, 2336, 7, 4428. The balanced split contains 85 training, 15 validation, and 15 reserved test images. See [dataset details](dataset.md).

All models use Adam with learning rate 0.001, batch size 16, and 30 epochs. The code resets the Torch seed to 47 for each model. The notebook metadata identifies a T4 runtime and the saved output reports CUDA; exact library versions are absent.

## Comparison at the selected checkpoints

| Model | Total parameters | Trainable parameters* | Selected epoch | Validation accuracy | Training seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Small CNN | 19,717 | 19,717 | 29 | 33.33% (5/15) | 18.9 |
| Deeper CNN | 389,701 | 389,701 | 28 | 53.33% (8/15) | 21.7 |
| Frozen ResNet18 + linear head | 11,179,077 | 2,565 | 30 | 80.00% (12/15) | 21.9 |

*Trainable parameter counts are derived from the architecture code; other table values are transcribed from the saved summary. Machine-readable data: [model_comparison.csv](../results/model_comparison.csv). Its test/inference fields are empty, not zero.

Small CNN: convolution widths 32/64 and dropout 0.25. Deeper CNN: 32/64/128/256 and dropout 0.35. Both use ReLU, max pooling, adaptive average pooling, and a five-class linear output. ResNet18 loads ImageNet weights, freezes the backbone, and trains a replacement 512-to-5 linear head. BatchNorm running statistics stay fixed.

## Checkpoint selection and timing

Each checkpoint is saved at the lowest validation loss. The selected checkpoint's validation accuracy is reported; the code does not select by peak validation accuracy.

Peak epoch accuracies are 33.3%, 60.0%, and 86.7%. Those are different statistics and must not replace the selected-checkpoint numbers. No held-out test accuracies have been measured in the supplied notebook.

Reported training time includes training, validation, checkpoint saving, printed epoch records, and history export. It excludes model construction and pretrained-weight loading. It is not inference latency.

## Preliminary model interpretation

ResNet18 leads the saved validation comparison at 80%, compared with 53.33% for the deeper CNN and 33.33% for the smaller CNN. This supports carrying ResNet18 forward as the leading candidate for the team to assess. It has only 2,565 trainable head parameters, which makes feature extraction practical for the small training set, while retaining a much larger total inference model.

The result rests on only 15 validation images, and inference speed has not been measured. This is a preliminary interpretation, not the team's finalized best-model justification. Use validation to guide selection; report final held-out test results without repeatedly tuning to the test set.

## Required additions

- Exact split manifest and verified checkpoint/class mapping.
- Held-out test accuracy for each architecture using its selected checkpoint.
- Per-class accuracy or confusion matrix for the chosen model, including class supports.
- Training/validation loss curves from all three history CSVs.
- Measured inference timing if used in the selection justification, with timing protocol.
- Final 1–2 paragraphs explaining the carry-forward decision.
- Exact environment, run/checkpoint references, and submission commit/tag.

The downloaded notebook contains rounded epoch accuracy printouts but no numeric loss history output. Full loss curves require the external CSVs written to Dario's Drive. Those files and the checkpoints are not in this repository.

Complete this document as the required 2–4-page Markdown/PDF results deliverable once those artifacts are available.
