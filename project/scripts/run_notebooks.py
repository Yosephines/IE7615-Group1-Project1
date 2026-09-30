"""Run the five notebooks under a new run ID, preserving the submitted notebooks."""
import argparse,os
from pathlib import Path
import nbformat
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run-id",required=True,help="New run name, e.g. ta_repro_01")
args=parser.parse_args()
if args.run_id == "group1_repro_v1":
    parser.error("The submitted run is protected. Choose a new run ID.")
if not args.run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.run_id):
    parser.error("Use letters, digits, underscores or hyphens for the run ID.")
os.environ.setdefault("CELEBA_DATA_DIR",str(ROOT.parent/"data/identities"))
os.environ["GROUP1_RUN_ID"]=args.run_id
os.environ.setdefault("GROUP1_DEVICE","cpu")
os.environ.setdefault("GROUP1_THREADS","2")
os.environ["JUPYTER_PATH"]=str(ROOT.parent/".venv/share/jupyter")+os.pathsep+os.environ.get("JUPYTER_PATH","")
output=ROOT.parent/"tmp/notebooks"/args.run_id
output.mkdir(parents=True,exist_ok=False)
for path in sorted((ROOT.parent/"notebooks").glob("0*.ipynb")):
    print("Executing",path.name,flush=True)
    notebook=nbformat.read(path,as_version=4)
    client=NotebookClient(notebook,timeout=1800,kernel_name="ie7615-group1",
                          resources={"metadata":{"path":str(ROOT.parent)}})
    try:
        client.execute()
    finally:
        nbformat.write(notebook,output/path.name)
    print("Completed",path.name,flush=True)
print("All five notebooks completed. Executed copies:",output,flush=True)
