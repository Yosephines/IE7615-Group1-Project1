# Checkpoints — awaiting handoff

Dario's notebook saves model_1.pt (small CNN), model_2.pt (deeper CNN), and model_3.pt (ResNet18) at the minimum validation loss for each run.

Each checkpoint contains `state_dict`, `ids`, and `model_number`. The supplied notebook does not embed these files, and they are not yet available in this repository.

Keep checkpoints in shared team storage or locally in this ignored folder. Record storage links and SHA-256 checksums in docs/training_results.md once obtained. Preserve class order and use the exact original split during evaluation.

The notebook also writes history CSVs to its checkpoint output directory; move copies of those CSVs to the tracked logs/ directory.
