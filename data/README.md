# Local dataset storage

Keep CelebA images and source identity annotations here locally. Dataset files in this folder are ignored by Git. Do not upload the dataset to the repository.

Suggested local layout: `data/raw/` for original images and annotations, `data/processed/` for derived data, and `data/splits/` for generated manifests. Document acquisition, filenames, split-generation seed, and exact regeneration steps in `docs/dataset.md`. Keep reproducible split-generation code in the data-preparation notebook or `src/`.
