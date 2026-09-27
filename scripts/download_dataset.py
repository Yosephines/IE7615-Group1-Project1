"""Restore the bundled subset from its recorded sources and verify exact bytes."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
cfg=json.loads((ROOT/"configs/dataset_source.json").read_text())
for row in cfg["inventory"]:
    target=ROOT/"data/identities"/row["file"]
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()==row["sha256"]:continue
    import gdown,kagglehub,shutil
    target.parent.mkdir(parents=True,exist_ok=True)
    try:
        gdown.download(id=row["drive_file_id"],output=str(target),quiet=True,
                       use_cookies=False,timeout=(10,20))
        if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest()!=row["sha256"]:
            raise ValueError("Drive download unavailable or different")
    except Exception:
        cached=kagglehub.dataset_download("jessicali9530/celeba-dataset/versions/2",
            path="img_align_celeba/img_align_celeba/"+target.name)
        shutil.copy2(cached,target)
    if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest()!=row["sha256"]:
        raise ValueError(f"Source bytes differ: {row['file']}")
print("All recorded images verified.")
