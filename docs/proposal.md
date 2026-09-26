# IE 7615 · Group 1 · Project 1 proposal

**Updated draft — team confirmation and one-page PDF export pending.**

## Objective and subset

Build a discriminative pipeline for celebrity identification and detection, beginning with single-face classification. The current baseline uses CelebA identity IDs 7007, 2970, 2336, 7, and 4428. It balances 23 images per identity, with 17 training, 3 validation, and 3 held-out test examples per class. Identity names and the broader selection rationale still require team confirmation.

## Framework and approach

The existing implementation uses PyTorch/torchvision in Google Colab. Dario has trained a small custom CNN, a deeper custom CNN, and an ImageNet-pretrained ResNet18 with a frozen backbone and trainable five-class head. The selected checkpoints achieve 33.33%, 53.33%, and 80.00% validation accuracy. Test evaluation remains pending.

ResNet18 is the leading validation candidate for the later detection milestones. The team will finalize its choice with documented trade-offs and report held-out test evaluation.

## Responsibilities

| Member | Contribution or assignment | Next milestone |
| --- | --- | --- |
| Dario Garza (TheMorrisGitHub) | Contributed the data preparation and three-model training/validation notebook | To be assigned by team |
| Yosephines | Repository owner; coordination of repository setup | To be assigned by team |
| Remaining team members | Names and evaluation/report responsibilities to be confirmed | To be assigned by team |

## Risks and planned mitigations

- Small evaluation sets: report class counts and avoid overstating accuracy differences.
- Split reproducibility: recover and version the exact split manifest before checkpoint evaluation.
- Missing handoff artifacts: collect history CSVs and share checkpoints with the team.
- Overfitting/data limitations: compare scratch and pretrained models using a shared protocol.
- Transition to detection: confirm annotation needs and compute requirements for Milestone 2.

## Items to confirm

Compare with the original Module 1 proposal to describe actual changes; that proposal was not available during this update. Confirm all team members, final identity rationale, division of labor, and the carry-forward model before exporting the final one-page PDF.
