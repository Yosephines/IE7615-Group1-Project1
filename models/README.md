# Saved models

The three selected checkpoints in `group1_repro_v1/` are included in this private repository. Each file contains a state dictionary, class order, architecture, selected epoch, run ID and split hash.

**Carry forward:** `group1_repro_v1/resnet18_frozen_backbone.pt`, epoch 28. See the [model handoff](../docs/best_model_handoff.md) for loading and preprocessing.

The two custom CNNs start from scratch. ResNet18 uses torchvision ImageNet weights with a frozen backbone and a trained five-class output layer. In raw checkpoint metadata, the `pretrained` argument is passed to all model constructors but has an effect only on ResNet18.

All files are below GitHub's 100 MB per-file limit. New model runs are ignored by default; intentionally add only the selected artifacts needed for a milestone.
