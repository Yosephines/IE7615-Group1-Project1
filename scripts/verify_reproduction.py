"""Repeat the fresh run on this machine and compare saved weights, logs and predictions."""
import contextlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from src.pipeline import Experiment,MODEL_NAMES,read_csv,write_json,sha256
RUN="group1_repro_v1"
reference=Experiment(ROOT/"data/identities",RUN)
replay_root=ROOT/"tmp/replay"
replay=Experiment(ROOT/"data/identities",RUN,replay_root,device="cpu")
replay.prepare()
checks=[]
for name in MODEL_NAMES:
    print("Reproducing",name,flush=True)
    log=replay_root/(name+"_stdout.txt");log.parent.mkdir(exist_ok=True,parents=True)
    with log.open("w") as h,contextlib.redirect_stdout(h):
        replay.train(name,epochs=30)
    new=replay.evaluate(name)
    old_path=reference.path("results",name+"_metrics.json")
    old=json.loads(old_path.read_text())
    # Independently reload original weights and regenerate evaluation.
    current=reference.evaluate(name,force=True)
    assert current["confusion_matrix"]==old["confusion_matrix"]
    assert read_csv(replay.path("logs",name+"_history.csv"))==read_csv(reference.path("logs",name+"_history.csv"))
    assert read_csv(replay.path("results",name+"_predictions.csv"))==read_csv(reference.path("results",name+"_predictions.csv"))
    a=torch.load(reference.path("models",name+".pt"),map_location="cpu",weights_only=True)["state_dict"]
    b=torch.load(replay.path("models",name+".pt"),map_location="cpu",weights_only=True)["state_dict"]
    assert a.keys()==b.keys() and all(torch.equal(a[k],b[k]) for k in a)
    checks.append({"architecture":name,"checkpoint_reload_predictions_match":True,
                   "repeated_training_histories_exact":True,"repeated_training_tensors_exact":True,
                   "repeated_test_predictions_exact":True})
    print("Verified exact repeat:",name,flush=True)
write_json(ROOT/"results"/RUN/"reproducibility_check.json",
    {"run_id":RUN,"device":"cpu","scope":"same machine, same recorded environment, two CPU threads",
     "manifest_sha256":sha256(reference.manifest),"checks":checks,
     "cross_machine_bitwise_equivalence_guaranteed":False})
