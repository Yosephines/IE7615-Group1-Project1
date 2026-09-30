# Run the project

Run these commands from the main repository folder in Windows PowerShell. Use Python 3.13.9. These instructions have been tested on Windows; other systems may need different package versions.

## 1. Install

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m ipykernel install --prefix .venv --name ie7615-group1 --display-name "IE7615 Group 1"
```

## 2. Check the saved models

```powershell
.venv\Scripts\python.exe project/scripts/verify_saved_run.py
```

This checks the included photos and models, then repeats the test predictions. It does not train or download anything.

## 3. Train again, if needed

```powershell
.venv\Scripts\python.exe project/scripts/run_notebooks.py --run-id my_run_01
```

This runs notebooks 01-05 in order. Use a new name after `--run-id` each time. The submitted results remain unchanged.

New files go into `project/configs/splits/`, `project/models/`, `project/logs/` and `project/results/`. Completed notebook copies go into `tmp/notebooks/my_run_01/`.

The first ResNet18 training run downloads its ImageNet starting weights if they are not already cached. Training takes several minutes on our CPU; other computers will differ.

## Update the reports

```powershell
.venv\Scripts\python.exe project/scripts/build_reports.py
.venv\Scripts\python.exe project/scripts/combine_reports.py
```

The first command rebuilds the results PDF and comparison table for the submitted experiment, `group1_repro_v1`. The second combines it with the team proposal. Neither changes the proposal.

Prediction times are kept as recorded. To measure them again, add `--measure-speed` to the first command.
