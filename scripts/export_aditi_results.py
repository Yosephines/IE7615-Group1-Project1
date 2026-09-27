"""Export recoverable evidence from Aditi's saved notebook; never invent loss values."""
import base64, csv, hashlib, json, re, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.pipeline import write_csv, write_json
source=ROOT/"notebooks"/"milestone01_team01 - outputtest.ipynb"
nb=json.loads(source.read_text(encoding="utf-8"))
texts=["".join(o.get("text",[])) for c in nb["cells"] for o in c.get("outputs",[])]
printed="\n".join(texts)
rows=[]
for number,name in enumerate(["small_custom_cnn","deeper_custom_cnn","resnet18_frozen_backbone"],1):
    matches=re.findall(rf"Model {number}, epoch (\d+): train ([\d.]+)%, val ([\d.]+)%",printed)
    assert [int(x[0]) for x in matches]==list(range(1,31))
    for epoch,train,val in matches:
        rows.append(dict(architecture=name,epoch=int(epoch),train_accuracy=float(train)/100,
                         val_accuracy=float(val)/100,precision="rounded notebook output"))
write_csv(ROOT/"logs/aditi_5id/rounded_epoch_accuracy.csv",rows)
summary="\n".join("".join(o.get("data",{}).get("text/plain",[]))
                  for o in nb["cells"][9]["outputs"])
summary_rows=re.findall(r"\d+\s+Model (\d)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)",summary)
assert len(summary_rows)==3
metrics=[]
names=["small_custom_cnn","deeper_custom_cnn","resnet18_frozen_backbone"]
for number,params,epoch,val,seconds in summary_rows:
    i=int(number)-1
    printed_acc=float(re.search(rf"Model {number} Test Accuracy: ([\d.]+)%",printed).group(1))
    correct=round(printed_acc*15/100)
    assert abs(correct/15*100-printed_acc)<.01
    metrics.append(dict(model=names[i],run_id="aditi_5id",selected_epoch=int(epoch),
        validation_accuracy=float(val),test_accuracy=correct/15,test_correct=correct,test_images=15,
        total_parameters=int(params),trainable_parameters=[19717,389701,2565][i],
        training_seconds=float(seconds),inference_ms_per_image="",
        provenance="saved Aditi notebook; trainable count derived from code"))
write_csv(ROOT/"results/aditi_5id/model_comparison.csv",metrics)
# Counts transcribed from the notebook's saved PNG and checked against its printed score.
cm=[[1,1,0,1,0],[0,2,0,0,1],[0,0,3,0,0],[0,0,0,3,0],[0,2,0,0,1]]
ids=["7007","2970","2336","7","4428"]
assert sum(map(sum,cm))==15 and sum(cm[i][i] for i in range(5))==metrics[2]["test_correct"]
write_csv(ROOT/"results/aditi_5id/resnet18_confusion.csv",
    [dict(true_identity=ids[i],**{identity:cm[i][j] for j,identity in enumerate(ids)}) for i in range(5)])
write_csv(ROOT/"results/aditi_5id/per_class_accuracy.csv",
    [dict(identity=identity,test_correct=cm[i][i],test_images=3,accuracy=cm[i][i]/3) for i,identity in enumerate(ids)])
fig=ROOT/"results/figures/aditi_5id";fig.mkdir(parents=True,exist_ok=True)
output=next(o["data"]["image/png"] for o in nb["cells"][17]["outputs"] if "image/png" in o.get("data",{}))
(fig/"resnet18_confusion.png").write_bytes(base64.b64decode(output))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
figure,axes=plt.subplots(1,3,figsize=(11,3.2),sharey=True)
for name,ax in zip(names,axes):
    selected=[r for r in rows if r["architecture"]==name]
    for split in ["train","val"]:
        ax.plot([r["epoch"] for r in selected],[r[split+"_accuracy"] for r in selected],label=split)
    ax.set(title=name.replace("_"," "),xlabel="Epoch",ylim=(0,1.02));ax.grid(alpha=.2)
axes[0].set_ylabel("Accuracy");axes[-1].legend()
figure.suptitle("Aditi's saved run: rounded epoch accuracy (not loss curves)",fontsize=11)
figure.tight_layout();figure.savefig(fig/"rounded_accuracy_curves.png",dpi=150);plt.close(figure)
write_json(ROOT/"results/aditi_5id/provenance.json",{
    "source_notebook":str(source.relative_to(ROOT)),"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
    "run_id":"aditi_5id","class_order":ids,"test_images":15,
    "confusion_source":"transcribed from saved PNG; row totals and trace checked",
    "epoch_accuracy_source":"rounded console output; not full-precision CSV histories",
    "loss_histories_available":(ROOT/"logs/aditi_5id/history_import.json").exists(),"exact_manifest_available":False,
    "checkpoint_files_available":False,"original_environment_versions_available":False})
print("Exported saved Aditi evidence; original notebooks unchanged.")
