"""Validate candidate original CSVs, then import and plot them.

Usage: python scripts/import_aditi_histories.py PATH_TO_ADITIS_OUTPUT_DIR
Requires the original run files, not histories from Dario or a new run.
"""
import argparse,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.pipeline import read_csv,write_csv,write_json,sha256,plot_history
parser=argparse.ArgumentParser();parser.add_argument("source",type=Path);args=parser.parse_args()
names=["small_custom_cnn","deeper_custom_cnn","resnet18_frozen_backbone"]
expected_epochs=[29,21,30];expected_acc=[5/15,8/15,12/15];collected=[]
for i,name in enumerate(names,1):
    path=args.source/f"model_{i}_history.csv"
    rows=read_csv(path)
    assert [int(r["epoch"]) for r in rows]==list(range(1,31)),f"{path}: wrong epochs"
    for r in rows:
        for key in ["train_loss","val_loss","train_accuracy","val_accuracy"]:
            value=float(r[key]);assert math.isfinite(value) and value>=0
            if "accuracy" in key:assert value<=1
    best=min(rows,key=lambda r:float(r["val_loss"]))
    assert int(best["epoch"])==expected_epochs[i-1],f"{path}: checkpoint epoch differs from Aditi"
    assert abs(float(best["val_accuracy"])-expected_acc[i-1])<1e-6
    rounded=read_csv(ROOT/"logs/aditi_5id/rounded_epoch_accuracy.csv")
    rounded=[r for r in rounded if r["architecture"]==name]
    for actual,displayed in zip(rows,rounded):
        for field in ["train_accuracy","val_accuracy"]:
            assert abs(float(actual[field])-float(displayed[field]))<=.000501,f"{path}: printed history mismatch"
    collected.append((name,path,rows))
for name,path,rows in collected:
    dest=ROOT/"logs/aditi_5id"/f"{name}_history.csv"
    if dest.exists() and read_csv(dest)!=rows:raise ValueError("Refusing to overwrite different imported history")
    write_csv(dest,rows)
    plot_history(dest,ROOT/"results/figures/aditi_5id"/f"{name}_curves.png")
write_json(ROOT/"logs/aditi_5id/history_import.json",{
    "validation":"All 30 epoch accuracies and selected checkpoint matched the saved notebook",
    "sources":{name:sha256(path) for name,path,_ in collected},
    "caution":"Matching histories do not verify checkpoint tensors or the exact image split."})
print("Validated original histories imported. Rebuild PDFs and update the submission audit.")
