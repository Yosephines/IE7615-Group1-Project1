# Results

`model_comparison.csv` is the current summary from **group1_repro_v1**. It includes validation/test accuracy, correct counts, parameter counts, training time and measured inference latency.

`group1_repro_v1/` contains per-model metrics, all 15 predictions per model, confusion matrices, training curves, the validation-based selection record and a full-repeat verification record. `saved_model_check.json` records independent checkpoint evaluation; `checkout_check.json` records the successful audit after Git checkout normalization. These files support the [current report](../docs/training_results.md).

`aditi_5id/` and `figures/aditi_5id/` preserve Aditi's earlier notebook results and genuine CSV-based curves. They are historical evidence and are not combined with the new run. `scripts/export_aditi_results.py` and `scripts/import_aditi_histories.py` rebuild those historical exports only.
