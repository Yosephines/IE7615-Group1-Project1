# Photos

`identities/` contains the 121 photos from the shared class collection. We use 23 per identity: 17 training, 3 validation and 3 test photos. Six photos are unused.

The identities and contributor credits are listed in the [main README](../README.md).

- [Image inventory](../project/configs/dataset_source.json): source locations and file checksums.
- [Saved split](../project/configs/splits/group1_repro_v1_seed47.csv): the exact 115 photos used for training, validation and testing.

All photos needed for this experiment are included. The saved models expect RGB images cropped to 224 x 224 pixels with ImageNet normalization; the notebooks apply this automatically.
