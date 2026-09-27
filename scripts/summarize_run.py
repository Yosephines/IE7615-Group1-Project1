"""Summarize the fresh run and measure batch-one forward latency on CPU."""
import json,statistics,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from src.pipeline import Experiment,MODEL_NAMES,IDS,make_model,make_loader,read_csv,write_csv,write_json
RUN="group1_repro_v1"
exp=Experiment(ROOT/"data/identities",RUN,device="cpu")
selected=json.loads(exp.path("results","selection.json").read_text())["selected_model"]
summary=[];classes=[]
for name in MODEL_NAMES:
    path=exp.path("results",name+"_metrics.json")
    result=json.loads(path.read_text())
    checkpoint=torch.load(exp.path("models",name+".pt"),map_location="cpu",weights_only=True)
    model=make_model(name,False).eval();model.load_state_dict(checkpoint["state_dict"])
    loader=make_loader(exp.data_dir,read_csv(exp.manifest),"test",1,47)
    images=[x for x,_,_ in loader]
    times=[]
    with torch.no_grad():
        for x in images:model(x)
        for _ in range(5):
            for x in images:
                t=time.perf_counter();model(x);times.append((time.perf_counter()-t)*1000)
    result["inference_ms_per_image"]=statistics.median(times)
    result["timing_protocol"]="CPU, 2 threads, batch size 1; one warm-up pass, median of 5 full test passes; forward pass only"
    result["hardware"]="Intel Core i7-7700K CPU @ 4.20 GHz"
    write_json(path,result)
    summary.append({k:result[k] for k in ["architecture","selected_epoch","validation_accuracy",
        "test_accuracy","test_correct","test_images","total_parameters","trainable_parameters",
        "training_seconds","inference_ms_per_image"]})
    for identity,accuracy in result["per_class_accuracy"].items():
        classes.append({"architecture":name,"identity":identity,"correct":round(accuracy*3),"test_images":3,"accuracy":accuracy})
write_csv(exp.path("results","model_comparison.csv"),summary)
write_csv(ROOT/"results/model_comparison.csv",summary)
write_csv(exp.path("results","per_class_accuracy.csv"),classes)
print(json.dumps({"selected_model":selected,"comparison":summary},indent=2))
