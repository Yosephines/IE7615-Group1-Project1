# Configuration and provenance

- `group1_repro_v1.json`: settings and artifact pointers for the current report.
- `dataset_source.json`: all 121 image names, source IDs and content hashes.
- `splits/group1_repro_v1_seed47.csv`: the exact 115-image training/validation/test split.
- `reproduction_environment.txt`: complete Python package versions from the actual Windows CPU run. Python was 3.13.9. Install with the PyTorch CPU extra index shown in the main README.
- `original_contributions.json`: byte-preservation hashes for the two original notebooks and three uploaded history CSVs.
- `baseline.json`: historical settings extracted from Dario's original notebook; retained as a reference.

Per-model `logs/group1_repro_v1/*_run.json` records the environment, pipeline hash, seed, parameter counts and selected checkpoint hash. The `pretrained` argument in these raw records only affects ResNet18; both custom CNNs are initialized from scratch.
