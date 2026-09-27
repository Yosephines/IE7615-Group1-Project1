# Dataset and split

We use five CelebA identity folders from the shared class dataset. IDs are the folder labels; celebrity names have not been independently verified. The folders were available to the team and each had enough images for a balanced experiment.

| Identity | Available | Train | Validation | Test | Unused |
| --- | --- | --- | --- | --- | --- |
| 7007 | 24 | 17 | 3 | 3 | 1 |
| 2970 | 25 | 17 | 3 | 3 | 2 |
| 2336 | 25 | 17 | 3 | 3 | 2 |
| 7 | 24 | 17 | 3 | 3 | 1 |
| 4428 | 23 | 17 | 3 | 3 | 0 |
| Total | 121 | 85 | 15 | 15 | 6 |

The output class order is **7007, 2970, 2336, 7, 4428**. The correct fifth ID is 4428, not the earlier introduction typo 4438.

## Original contributors

These identities come from the shared class collection. The class claim sheet supplied by Yosephine lists the following contributors; they are not the membership list for this project team.

| Identity ID | Claimed by | Group |
| --- | --- | --- |
| 7007 | Mus Ab Irfan Yilmaz | 3 |
| 2970 | Rhea Paul | 3 |
| 2336 | Masato Kan | 3 |
| 7 | Jin-woo Hong | 2 |
| 4428 | David Fung | 4 |

## Sources

Images are included under `data/identities`. The [source inventory](../configs/dataset_source.json) records each file, identity, SHA-256, Drive file ID and acquisition source.

- [Shared class Drive](https://drive.google.com/drive/folders/1oFdEmVjo5uy-xsM2fNscKQQKhe8pU30P): identity assignments and filenames.
- [CelebA mirror, version 2](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset/versions/2): exact filenames used when Drive downloads stalled. Reference image `005770.jpg` was compared across sources and matched byte for byte.

All 121 images were decoded and checked for duplicate content. Their recorded hashes now fix the data version. This is a small private course subset, not a redistribution of the full dataset.

## How the split is made

Sort filenames within each folder, shuffle with `random.Random(47 + identity)`, then retain 23 per identity. The first 17 are training, the next 3 validation, and the last 3 test. The resulting ratio is 73.9% / 13.0% / 13.0%; six remaining images are unused.

The [manifest](../configs/splits/group1_repro_v1_seed47.csv) records every selected filename, identity, split and hash. The pipeline rejects duplicate bytes, missing files, changed content and an attempt to replace an existing split. All architectures use this one manifest. Byte checks do not rule out visually near-duplicate photos.

## Preprocessing

Convert to RGB. Training uses `RandomResizedCrop(224, scale=(0.85, 1.0))` and random horizontal flips. Validation/test use resize-to-256 on the shorter edge followed by a 224 center crop. Normalize with mean `(0.485, 0.456, 0.406)` and standard deviation `(0.229, 0.224, 0.225)`.

The new run creates its own complete record. It does not claim to reconstruct Aditi's original file ordering or training environment.
