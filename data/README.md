# Local dataset

The current notebook reads a Google Drive DATA_DIR containing these identity folders:

```text
DATA_DIR/
  7007/
  2970/
  2336/
  7/
  4428/
```

Each contains jpg/jpeg/png images. Point DATA_DIR to your own dataset location when running in Colab. See [dataset notes](../docs/dataset.md) for the saved counts and preprocessing.

This repository does not contain images. Everything under this local data/ directory except this README is ignored. Store the recovered split manifest under configs/splits/ so it can be versioned without image files.
