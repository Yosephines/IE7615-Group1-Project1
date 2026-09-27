# Run and check the project

Run the following commands from the repository folder. They use Windows PowerShell and Python 3.13.9, the version used for our results. On macOS/Linux, replace `.venv\Scripts\python.exe` with `.venv/bin/python`.

## Install the software

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m ipykernel install --prefix .venv --name ie7615-group1 --display-name "IE7615 Group 1"
```

The complete list of installed versions is in `configs/reproduction_environment.txt`. To install that exact Windows package list, use `pip install -r configs/reproduction_environment.txt --extra-index-url https://download.pytorch.org/whl/cpu`. Other operating systems may require different packages.

## Check the saved models

```powershell
.venv\Scripts\python.exe scripts/verify_saved_run.py
```

This checks that the photos and models have not changed, loads the saved models and repeats their test predictions. It does not retrain or download model weights.

## Train all three models again

```powershell
.venv\Scripts\python.exe scripts/run_notebooks.py --run-id ta_repro_01
```

Use a new name after `--run-id` each time. New models, logs and results are saved under that name. Completed notebook copies go to `tmp/notebooks/ta_repro_01`, leaving the submitted notebooks unchanged.

The first ResNet18 training run downloads its ImageNet starting weights if they are not already cached. Training all three models took about six minutes on our Intel Core i7-7700K CPU. Other computers will differ.

## Additional checks

- `python scripts/audit_submission.py` checks the notebooks, reports, file links and saved models.
- `python scripts/verify_reproduction.py` trains again and compares every training result, saved model value and prediction. It writes to `tmp/replay`; move any previous replay folder before repeating this check.
- `python scripts/build_reports.py` rebuilds the two PDF reports and their Markdown copies from the submitted results.
- `python scripts/summarize_run.py` updates the comparison table and remeasures prediction speed. If you run it, rebuild the reports afterward so timings agree.

Use the project Python environment for these commands. The report scripts use the submitted experiment, `group1_repro_v1`. See the [README](../README.md) for the notebook order.
