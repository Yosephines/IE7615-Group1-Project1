# Training histories — awaiting handoff

Dario's notebook writes model_1_history.csv, model_2_history.csv, and model_3_history.csv to its Google Drive OUTPUT_DIR. These files have not yet been added to the repository.

Place copies in `logs/dario_baseline/`. Columns written by the notebook:

```text
epoch,train_loss,train_accuracy,val_loss,val_accuracy
```

Accuracies are fractions from 0 to 1. Use these CSVs to generate loss/accuracy plots under `results/figures/`. The saved notebook printouts do not contain loss values and are not substitutes for the original CSVs.

Keep histories here rather than under models/: checkpoint-directory contents are ignored by Git. Record exact dependency versions and hardware alongside the handed-off artifacts.
