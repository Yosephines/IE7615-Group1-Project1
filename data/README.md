# Data

`identities/` contains the 121 images used for the fresh Group 1 experiment. The split uses 115 of them; six are unused after balancing.

See [dataset notes](../docs/dataset.md), [source inventory](../configs/dataset_source.json) and [split manifest](../configs/splits/group1_repro_v1_seed47.csv).

No download is needed after cloning this version. If a file is missing, `python scripts/download_dataset.py` restores the recorded subset and checks exact hashes; it requires network access and the original sources to remain available.

Keep future large datasets outside Git and document their versions, storage locations and checksums. The current small fixed subset is intentionally included for TA review.
