"""Verify bundled images and independently evaluate all three saved checkpoints."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from PIL import Image
from src.pipeline import Experiment,IDS,MODEL_NAMES,sha256,validate_split,read_csv,make_model,make_loader,write_json
RUN="group1_repro_v1"
exp=Experiment(ROOT/"data/identities",RUN,device="cpu")
source=json.loads((ROOT/"configs/dataset_source.json").read_text())
seen=set()
for row in source["inventory"]:
    file=exp.data_dir/row["file"]
    digest=sha256(file)
    assert digest==row["sha256"] and digest not in seen, f"Changed/duplicate image: {file}"
    seen.add(digest)
    with Image.open(file) as image:image.verify()
rows=validate_split(exp.data_dir,exp.manifest)
assert len(rows)==115 and len(seen)==121
for identity in IDS:
    for split,count in [("train",17),("val",3),("test",3)]:
        assert sum(r["identity"]==identity and r["split"]==split for r in rows)==count
checks=[]
for name in MODEL_NAMES:
    m=json.loads(exp.path("results",name+"_metrics.json").read_text())
    ckpt=exp.path("models",name+".pt")
    assert sha256(ckpt)==m["checkpoint_sha256"]
    assert sha256(ROOT/"src/pipeline.py")==m["pipeline_sha256"]
    saved=torch.load(ckpt,map_location="cpu",weights_only=True)
    assert saved["ids"]==IDS and saved["architecture"]==name
    assert saved["manifest_sha256"]==sha256(exp.manifest)==m["manifest_sha256"]
    history=read_csv(exp.path("logs",name+"_history.csv"))
    assert [int(r["epoch"]) for r in history]==list(range(1,31))
    best=min(history,key=lambda r:float(r["val_loss"]))
    assert int(best["epoch"])==saved["epoch"]==m["selected_epoch"]
    assert float(best["val_accuracy"])==m["validation_accuracy"]
    model=make_model(name,False).eval();model.load_state_dict(saved["state_dict"])
    assert sum(p.numel() for p in model.parameters())==m["total_parameters"]
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==m["trainable_parameters"]
    predictions=[];cm=[[0]*5 for _ in IDS]
    with torch.no_grad():
        for x,y,files in make_loader(exp.data_dir,rows,"test",16,47):
            guesses=model(x).argmax(1).tolist()
            for file,actual,guess in zip(files,y.tolist(),guesses):
                cm[actual][guess]+=1
                predictions.append(dict(file=file,true_identity=IDS[actual],predicted_identity=IDS[guess],correct=str(int(actual==guess))))
    assert predictions==read_csv(exp.path("results",name+"_predictions.csv"))
    assert cm==m["confusion_matrix"]
    correct=sum(int(r["correct"]) for r in predictions)
    assert correct==m["test_correct"] and correct/len(predictions)==m["test_accuracy"]
    print(f"Verified {name}: {correct}/{len(predictions)} correct")
    checks.append({"architecture":name,"checkpoint_sha256":sha256(ckpt),"test_correct":correct,"predictions_match":True})
selection=json.loads(exp.path("results","selection.json").read_text())
records=[json.loads(exp.path("logs",n+"_run.json").read_text()) for n in MODEL_NAMES]
assert selection["selected_model"]==max(records,key=lambda r:r["validation_accuracy"])["architecture"]
write_json(exp.path("results","saved_model_check.json"),{"run_id":RUN,"images_verified":121,"split_images":115,"model_checks":checks})
print("All source images, split, histories, checkpoint hashes and fresh predictions verified.")
