# Milestone 1 submission audit

## Local deliverables

| Requirement | Evidence |
| --- | --- |
| 4â€“6 identities and rationale | Five IDs, available counts and balance rationale in `docs/dataset.md` |
| Data-preparation notebook | Executed `notebooks/01_data_preparation.ipynb` |
| One training notebook per architecture | Executed notebooks 02, 03 and 04 |
| Held-out test comparison | Notebook 05 and `results/model_comparison.csv` |
| Per-class results | ResNet18 confusion matrix and `results/group1_repro_v1/per_class_accuracy.csv` |
| Training curves | Three full 30-epoch histories and genuine loss plots |
| Model size/time/accuracy | Comparison CSV and three-page training report; inference also measured |
| Best-model justification | Results report and `docs/best_model_handoff.md` |
| Refreshed one-page proposal | `output/pdf/group1_proposal.pdf` |
| 2â€“4-page results report | `output/pdf/group1_training_results.pdf` (three pages) |
| Reproduction instructions | Root README, pinned dependencies, saved data/split/models |
| Later milestone continuity | ResNet18 checkpoint and `docs/roadmap.md` |

## Checks performed

- All 121 source images decode and match their recorded SHA-256; no identical image content is repeated.
- The 115-image manifest contains 17/3/3 images per identity, with disjoint paths and hashes. Six images remain unused.
- All five numbered notebooks executed on the actual data without cell errors.
- All three models trained for 30 epochs. Their checkpoints match minimum validation loss; architecture selection used validation before the test evaluation.
- Saved model files were independently reloaded and all 45 test predictions matched their saved records.
- A second complete training run matched all histories, checkpoint tensors and test predictions exactly in the same recorded environment.
- Four pipeline integration tests passed, covering the models, checkpoint behavior and invalid-data rejection.
- Original Dario/Aditi notebooks and Aditi's uploaded history CSVs retain their source bytes. Their provenance is recorded in `configs/original_contributions.json`.
- Git attributes preserve the byte-sensitive split, source code and original evidence; data and checkpoint files are included rather than ignored.
- The full artifact audit also passed in a separate Git-normalized checkout using a temporary index. All 121 images and three checkpoints were present, and the real staging index was unchanged. See `results/group1_repro_v1/checkout_check.json`.
- The proposal is one page and the report is three pages; every rendered page was visually checked.

Run `python scripts/audit_submission.py` to recheck notebook execution, file links, reports, source evidence and data/model integrity. `results/group1_repro_v1/reproducibility_check.json` records the full training repeat; the audit does not retrain.

## Limits and final submission actions

The test set has only 15 images (three per identity); one image changes overall accuracy by 6.7 percentage points. Exact retraining was verified on this machine and environment, not every hardware platform. Identity IDs are the class dataset labels; names have not been independently verified.

For Canvas, include the submission commit hash or release tag and confirm that the private repository gives the team and teaching staff read-write access. Staff invitation acceptance has not been verified.
