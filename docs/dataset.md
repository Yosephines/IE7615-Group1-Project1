# Milestone 1 dataset

Source of counts and settings: saved outputs and code in [Dario's notebook](../notebooks/milestone01_team01_dario.ipynb). Dataset images were not independently inspected during repository cleanup.

## Identity subset

| Class index | CelebA identity ID | Available images | Selected | Training | Validation | Test |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 7007 | 24 | 23 | 17 | 3 | 3 |
| 1 | 2970 | 25 | 23 | 17 | 3 | 3 |
| 2 | 2336 | 25 | 23 | 17 | 3 | 3 |
| 3 | 7 | 24 | 23 | 17 | 3 | 3 |
| 4 | 4428 | 23 | 23 | 17 | 3 | 3 |
| **Total** | | **121** | **115** | **85** | **15** | **15** |

The notebook selects five available labeled folders, checks a minimum of 20 images each, and balances classes to the smallest folder count. Six available images are excluded. Celebrity names and visual-diversity claims have not been verified. Document label provenance against the original CelebA identity annotation before final submission.

The original introduction's 4438 was a typo. The code, saved outputs, and shared Drive folder listing use 4428; only the introduction was corrected.

## Split procedure

For each identity, the notebook sorts the relative image paths, shuffles with `random.Random(47 + int(identity))`, selects 23 images, then partitions them. Validation/test counts use `round(23 * 0.15)` and training receives the remainder.

The intended proportions are 70/15/15; actual proportions are about 73.9/13.0/13.0 because of rounding. All three models share the same in-memory manifest.

**Missing artifact:** despite mentioning a saved split.csv, the current notebook never writes the manifest. Obtain the original runtime manifest or recover it from the unchanged original file inventory. Store the exact manifest at `configs/splits/milestone1_seed47.csv` and load it in subsequent notebooks. That path is reserved; the file has not been supplied.

Check duplicate paths and image-content duplicates across splits. Split generation by list slicing separates entries but does not detect duplicate image content under different filenames.

## Preprocessing

All images are converted to RGB.

- Training: RandomResizedCrop(224, scale=(0.85, 1.0)), random horizontal flip, tensor conversion, normalization.
- Validation: resize shorter edge to 256, center crop 224, tensor conversion, normalization.
- Mean: [0.485, 0.456, 0.406]; standard deviation: [0.229, 0.224, 0.225].
- Planned test evaluation should reuse deterministic validation transforms; no test loader currently exists.

## Reproduction and limits

The notebook reads jpg/jpeg/png files recursively from identity-named folders in Google Drive. Team members must configure their own DATA_DIR; folder contents must match the original inventory to reproduce the split.

Only 15 validation and 15 test examples exist. One prediction changes overall accuracy by 6.67 percentage points; each class has three evaluation examples. Report these counts alongside results.

Still needed: dataset source/version and annotation provenance, exact shared manifest, duplicate checks, and the team's fuller identity-selection rationale.
