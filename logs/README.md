# Training logs

## Current report: group1_repro_v1

Each architecture has a 30-row `*_history.csv` with full-precision training/validation loss and accuracy, plus a `*_run.json` containing settings, timing, environment and checkpoint hash. These are actual executed runs, not reconstructed curves.

## Historical: aditi_5id

`original/model_1_history.csv`, `model_2_history.csv` and `model_3_history.csv` preserve the uploaded CSV bytes from commit `c0f5920`. Named history files normalize their columns; `history_import.json` records matching and hashes. `rounded_epoch_accuracy.csv` is an older notebook-output extract, retained as supporting evidence.

Historical logs do not supply the current report's numbers. New experiments use a new run folder.
