# Notebook handoff

## Current implementation

[milestone01_team01_dario.ipynb](milestone01_team01_dario.ipynb) is Dario Garza's combined data-preparation, three-model training, and validation notebook. Saved outputs are retained. Its introduction now correctly lists identity 4428.

The empty scaffold notebooks were removed to avoid suggesting separate implementations already exist.

## What Dario needs to share

The notebook writes these external files to its configured OUTPUT_DIR in Drive:

- model_1.pt, model_2.pt, model_3.pt: minimum-validation-loss checkpoints.
- model_1_history.csv, model_2_history.csv, model_3_history.csv: full epoch loss/accuracy histories.

Copy histories to `logs/dario_baseline/`; keep checkpoints in shared team storage. Add storage links and checksums to the results document once available.

The notebook mentions split.csv but does not save it. Export the original in-memory `manifest`, if available, to `configs/splits/milestone1_seed47.csv`. Otherwise recover it only after confirming the original file inventory is unchanged. Do not pair a different regenerated split with the old checkpoints.

After defining REPO_ROOT as the cloned repository path:

```python
split_dir = REPO_ROOT / "configs" / "splits"
split_dir.mkdir(parents=True, exist_ok=True)
assert not manifest["file"].duplicated().any()
manifest.to_csv(split_dir / "milestone1_seed47.csv", index=False)
```

Read identity values as strings when loading the CSV. Preserve class order from the checkpoint's `ids` field. Check duplicate image content separately.

## Required final organization

The assignment requires a data-preparation notebook and one training notebook per compared architecture. The current combined notebook does not yet meet that packaging requirement. When refactoring:

| Final notebook | Content from current notebook |
| --- | --- |
| 01_data_preparation.ipynb | Identity checks, split export, class mapping, transforms |
| 02_custom_cnn.ipynb | Small CNN (Model 1) |
| 03_transfer_learning.ipynb | Frozen ResNet18 (Model 3) |
| 04_deeper_cnn.ipynb | Deeper CNN (Model 2) |

Make each training notebook load the exact shared split and train only its intended architecture. Extract shared dataset/model/training functions into `src/` when implementing this refactor. A source folder has not been kept solely as a placeholder.

Keep this original as the recorded baseline until the new notebooks are verified. Refactoring code does not establish that new outputs were produced; retain clear provenance for the existing run.

## Final evaluation

Load each saved checkpoint into the matching architecture, use evaluation mode and deterministic transforms, and evaluate the reserved test split. Do not evaluate the last in-memory training model as a substitute for the selected checkpoint.
