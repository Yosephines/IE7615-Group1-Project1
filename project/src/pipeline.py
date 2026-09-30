"""Group 1 five-identity pipeline, refactored from Dario and Aditi's notebooks.

New executions produce separately named runs. They do not reproduce or replace
Aditi's historical metrics without the original dataset, split and environment.
"""
import csv
import hashlib
import json
import os
import platform
import random
import time
from pathlib import Path

IDS = ["7007", "2970", "2336", "7", "4428"]
MODEL_NAMES = ["small_custom_cnn", "deeper_custom_cnn", "resnet18_frozen_backbone"]
ROOT = Path(__file__).resolve().parents[1]

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot export empty rows")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))

def prepare_split(data_dir, destination, seed=47, minimum=20):
    """Save a content-checked balanced split; refuse to replace a different split."""
    data_dir, destination = Path(data_dir).resolve(), Path(destination)
    inventory, seen = {}, {}
    for identity in IDS:
        folder = data_dir / identity
        paths = sorted(p for p in folder.rglob("*")
                       if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if len(paths) < minimum:
            raise ValueError(f"{identity}: {len(paths)} images; need at least {minimum}")
        for p in paths:
            digest = sha256(p)
            if digest in seen:
                raise ValueError(f"Duplicate image content: {seen[digest]} and {p}")
            seen[digest] = str(p)
        inventory[identity] = paths
    n = min(map(len, inventory.values()))
    n_val = n_test = round(n * .15)
    n_train = n - n_val - n_test
    rows = []
    for identity, paths in inventory.items():
        random.Random(seed + int(identity)).shuffle(paths)
        for i, p in enumerate(paths[:n]):
            split = "train" if i < n_train else "val" if i < n_train+n_val else "test"
            rows.append(dict(identity=identity, split=split,
                             file=p.relative_to(data_dir).as_posix(), sha256=sha256(p)))
    if destination.exists():
        if read_csv(destination) != rows:
            raise ValueError("Existing split differs. Choose a new run ID; do not overwrite it.")
    else:
        write_csv(destination, rows)
    validate_split(data_dir, destination)
    return {"available": {k: len(v) for k,v in inventory.items()},
            "selected_per_identity": n, "train_per_identity": n_train,
            "val_per_identity": n_val, "test_per_identity": n_test,
            "manifest_sha256": sha256(destination)}

def validate_split(data_dir, manifest):
    root = Path(data_dir).resolve()
    rows = read_csv(manifest)
    if not rows or set(r["identity"] for r in rows) != set(IDS):
        raise ValueError("Manifest must contain exactly the five Group 1 identities")
    paths, hashes = set(), set()
    for row in rows:
        relative = Path(row["file"])
        p = (root / relative).resolve()
        if relative.is_absolute() or not p.is_relative_to(root):
            raise ValueError("Manifest path escapes dataset directory")
        if relative.parts[0] != row["identity"] or row["split"] not in {"train","val","test"}:
            raise ValueError("Invalid identity folder or split")
        digest = sha256(p)
        if digest != row["sha256"] or p in paths or digest in hashes:
            raise ValueError("Changed or duplicate image in split")
        paths.add(p); hashes.add(digest)
    for identity in IDS:
        for split in ("train","val","test"):
            if not any(r["identity"] == identity and r["split"] == split for r in rows):
                raise ValueError(f"Empty {identity}/{split}")
    return rows

def environment():
    import torch, torchvision, pandas, matplotlib, PIL
    return {"python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "torchvision": torchvision.__version__,
            "pandas": pandas.__version__, "matplotlib": matplotlib.__version__,
            "pillow": PIL.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}

def make_model(name, pretrained=True):
    import torch
    from torch import nn
    from torchvision import models
    if name == "resnet18_frozen_backbone":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        for p in model.parameters():
            p.requires_grad = False
        model.fc = nn.Linear(512, len(IDS))
        return model
    settings = {"small_custom_cnn": ([32,64],.25),
                "deeper_custom_cnn": ([32,64,128,256],.35)}
    if name not in settings:
        raise ValueError(name)
    widths, dropout = settings[name]
    layers, previous = [], 3
    for width in widths:
        layers.extend([nn.Conv2d(previous,width,3,padding=1),nn.ReLU(),nn.MaxPool2d(2)])
        previous = width
    # Keep original state_dict key names for compatibility with the supplied notebooks.
    model = nn.Module()
    model.network = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(dropout), nn.Linear(previous,len(IDS)))
    model.forward = model.network.forward
    return model

def make_loader(data_dir, rows, split, batch_size, seed):
    import torch
    from PIL import Image
    from torchvision import transforms
    ops = ([transforms.RandomResizedCrop(224,scale=(.85,1.0)),
            transforms.RandomHorizontalFlip()] if split == "train" else
           [transforms.Resize(256),transforms.CenterCrop(224)])
    transform = transforms.Compose(ops + [transforms.ToTensor(),
        transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
    selected = [r for r in rows if r["split"] == split]
    class Faces(torch.utils.data.Dataset):
        def __len__(self): return len(selected)
        def __getitem__(self,index):
            row = selected[index]
            with Image.open(Path(data_dir)/row["file"]) as image:
                x = transform(image.convert("RGB"))
            return x, IDS.index(row["identity"]), row["file"]
    return torch.utils.data.DataLoader(Faces(), batch_size=batch_size,
        shuffle=(split=="train"), num_workers=0,
        generator=torch.Generator().manual_seed(seed))

def run_epoch(model, loader, device, optimizer=None, frozen=False):
    import torch
    model.train(optimizer is not None)
    if frozen: model.eval()
    loss_sum=correct=total=0
    with torch.set_grad_enabled(optimizer is not None):
        for x,y,_ in loader:
            x,y=x.to(device),y.to(device)
            if optimizer is not None: optimizer.zero_grad()
            logits=model(x)
            loss=torch.nn.functional.cross_entropy(logits,y)
            if optimizer is not None:
                loss.backward();optimizer.step()
            loss_sum+=loss.item()*len(y)
            correct+=(logits.argmax(1)==y).sum().item()
            total+=len(y)
    return loss_sum/total,correct/total

class Experiment:
    def __init__(self,data_dir,run_id="group1_repro_v1",root=ROOT,device=None):
        import torch
        if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in run_id):
            raise ValueError("Use letters, digits, underscores or hyphens for run_id")
        self.root=Path(root);self.data_dir=Path(data_dir);self.run_id=run_id
        self.manifest=self.root/"configs"/"splits"/f"{run_id}_seed47.csv"
        self.device=torch.device(device or os.environ.get("GROUP1_DEVICE", "cpu"))
        torch.set_num_threads(int(os.environ.get("GROUP1_THREADS", "2")))
        torch.use_deterministic_algorithms(True)

    def path(self,kind,name):
        return self.root/kind/self.run_id/name

    def prepare(self):
        return prepare_split(self.data_dir,self.manifest)

    def train(self,name,epochs=30,pretrained=True):
        import torch
        if epochs < 1: raise ValueError("epochs must be positive")
        checkpoint=self.path("models",name+".pt")
        if checkpoint.exists():
            raise FileExistsError("Checkpoint already exists. Choose a new run ID to retrain.")
        rows=validate_split(self.data_dir,self.manifest)
        random.seed(47);torch.manual_seed(47)
        model=make_model(name,pretrained).to(self.device)
        optimizer=torch.optim.Adam((p for p in model.parameters() if p.requires_grad),lr=.001)
        train=make_loader(self.data_dir,rows,"train",16,47)
        val=make_loader(self.data_dir,rows,"val",16,47)
        checkpoint.parent.mkdir(parents=True,exist_ok=True)
        best=float("inf");history=[]
        start=time.perf_counter()
        for epoch in range(1,epochs+1):
            tl,ta=run_epoch(model,train,self.device,optimizer,name=="resnet18_frozen_backbone")
            vl,va=run_epoch(model,val,self.device)
            history.append(dict(epoch=epoch,train_loss=tl,train_accuracy=ta,val_loss=vl,val_accuracy=va))
            if vl < best:
                best=vl
                torch.save({"state_dict":model.state_dict(),"ids":IDS,"architecture":name,
                    "epoch":epoch,"manifest_sha256":sha256(self.manifest),
                    "run_id":self.run_id,"pretrained":pretrained},checkpoint)
            print(f"{name} epoch {epoch}: loss {tl:.4f}/{vl:.4f}, accuracy {ta:.3f}/{va:.3f}")
        seconds=time.perf_counter()-start
        write_csv(self.path("logs",name+"_history.csv"),history)
        selected=min(history,key=lambda r:r["val_loss"])
        metadata={"run_id":self.run_id,"architecture":name,"epochs":epochs,"seed":47,
            "batch_size":16,"learning_rate":.001,"ids":IDS,"device":str(self.device),
            "pretrained":pretrained,"selected_epoch":selected["epoch"],
            "validation_accuracy":selected["val_accuracy"],"training_seconds":seconds,
            "total_parameters":sum(p.numel() for p in model.parameters()),
            "trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),
            "manifest_sha256":sha256(self.manifest),"checkpoint_sha256":sha256(checkpoint),
            "environment":environment(), "cpu_threads":torch.get_num_threads(),
            "deterministic_algorithms":torch.are_deterministic_algorithms_enabled(),
            "pipeline_sha256":sha256(Path(__file__)),
            "preprocessing":{"resize":256,"crop":224,"mean":[.485,.456,.406],"std":[.229,.224,.225]}}
        write_json(self.path("logs",name+"_run.json"),metadata)
        plot_history(self.path("logs",name+"_history.csv"),
                     self.path("results",name+"_curves.png"))
        return metadata

    def evaluate(self,name,force=False):
        import torch
        rows=validate_split(self.data_dir,self.manifest)
        checkpoint=self.path("models",name+".pt")
        saved=torch.load(checkpoint,map_location=self.device,weights_only=True)
        if saved["ids"]!=IDS or saved["architecture"]!=name or saved["manifest_sha256"]!=sha256(self.manifest):
            raise ValueError("Checkpoint identity/architecture/split mismatch")
        result_path=self.path("results",name+"_metrics.json")
        if result_path.exists() and not force:
            cached=json.loads(result_path.read_text())
            if cached["checkpoint_sha256"]!=sha256(checkpoint):
                raise ValueError("Checkpoint changed after evaluation")
            return cached
        model=make_model(name,False).to(self.device)
        model.load_state_dict(saved["state_dict"]);model.eval()
        cm=[[0]*len(IDS) for _ in IDS]; predictions=[]
        with torch.no_grad():
            for x,y,files in make_loader(self.data_dir,rows,"test",16,47):
                pred=model(x.to(self.device)).argmax(1).cpu().tolist()
                for file,actual,guess in zip(files,y.tolist(),pred):
                    cm[actual][guess]+=1
                    predictions.append(dict(file=file,true_identity=IDS[actual],
                                            predicted_identity=IDS[guess],correct=int(actual==guess)))
        result=json.loads(self.path("logs",name+"_run.json").read_text())
        if result["checkpoint_sha256"] != sha256(checkpoint):
            raise ValueError("Checkpoint differs from training record")
        result.update(test_images=len(predictions),test_correct=sum(r["correct"] for r in predictions),
            test_accuracy=sum(r["correct"] for r in predictions)/len(predictions),
            per_class_accuracy={identity:cm[i][i]/sum(cm[i]) for i,identity in enumerate(IDS)},
            confusion_matrix=cm)
        write_csv(self.path("results",name+"_predictions.csv"),predictions)
        write_json(result_path,result)
        plot_confusion(cm,self.path("results",name+"_confusion.png"))
        return result

def plot_history(history_path,destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows=read_csv(history_path)
    fig,axes=plt.subplots(1,2,figsize=(10,3.6))
    for ax,metric in zip(axes,["loss","accuracy"]):
        for split in ["train","val"]:
            ax.plot([int(r["epoch"]) for r in rows],
                    [float(r[split+"_"+metric]) for r in rows],label=split)
        ax.set(xlabel="Epoch",ylabel=metric.capitalize());ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();Path(destination).parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(destination,dpi=150);plt.close(fig)

def plot_confusion(cm,destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(6,5))
    ax.imshow(cm,cmap="Blues")
    for i in range(len(IDS)):
        for j in range(len(IDS)):
            ax.text(j,i,str(cm[i][j]),ha="center",va="center",color="white" if cm[i][j]>1.5 else "black")
    ax.set_xticks(range(len(IDS)),IDS);ax.set_yticks(range(len(IDS)),IDS)
    ax.set(xlabel="Predicted identity",ylabel="True identity",title="Held-out test confusion matrix")
    fig.tight_layout();Path(destination).parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(destination,dpi=150);plt.close(fig)
