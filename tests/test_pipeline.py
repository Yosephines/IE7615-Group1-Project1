"""Integration tests use generated images only; no synthetic metrics enter results/."""
import csv,json,tempfile,unittest,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from PIL import Image
from src.pipeline import IDS,MODEL_NAMES,Experiment,make_model,prepare_split,validate_split,read_csv,sha256
torch.set_num_threads(2)

class PipelineIntegration(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name);self.data=self.root/"data"
        rng=np.random.default_rng(123)
        for identity in IDS:
            folder=self.data/identity;folder.mkdir(parents=True)
            for i in range(23):
                Image.fromarray(rng.integers(0,256,(48,48,3),dtype=np.uint8)).save(folder/f"{i:03d}.png")
        self.exp=Experiment(self.data,"synthetic_test",self.root,device="cpu")
    def tearDown(self):self.temp.cleanup()

    def test_split_roundtrip_and_mutation_detection(self):
        self.exp.prepare();before=self.exp.manifest.read_bytes()
        self.exp.prepare();self.assertEqual(before,self.exp.manifest.read_bytes())
        rows=read_csv(self.exp.manifest)
        self.assertEqual({s:sum(r["split"]==s for r in rows) for s in ["train","val","test"]},{"train":85,"val":15,"test":15})
        p=self.data/rows[0]["file"];p.write_bytes(b"changed")
        with self.assertRaises(ValueError):validate_split(self.data,self.exp.manifest)

    def test_duplicate_detection(self):
        (self.data/IDS[1]/"000.png").write_bytes((self.data/IDS[0]/"000.png").read_bytes())
        with self.assertRaisesRegex(ValueError,"Duplicate"):self.exp.prepare()

    def test_all_architectures_train_reload_evaluate(self):
        self.exp.prepare()
        for name,total,trainable in zip(MODEL_NAMES,[19717,389701,11179077],[19717,389701,2565]):
            # ResNet uses random frozen weights only in this synthetic smoke test: no download.
            record=self.exp.train(name,epochs=1,pretrained=False)
            self.assertEqual(record["total_parameters"],total)
            self.assertEqual(record["trainable_parameters"],trainable)
            result=self.exp.evaluate(name)
            self.assertEqual(result["test_images"],15)
            self.assertEqual(sum(map(sum,result["confusion_matrix"])),15)
            self.assertEqual(result,self.exp.evaluate(name))
            self.assertTrue(self.exp.path("results",name+"_curves.png").exists())
            with self.assertRaises(FileExistsError):self.exp.train(name,epochs=1,pretrained=False)
        # A checkpoint with a different label order must never be accepted.
        p=self.exp.path("models",MODEL_NAMES[0]+".pt")
        saved=torch.load(p,weights_only=True);saved["ids"]=list(reversed(IDS));torch.save(saved,p)
        with self.assertRaises(ValueError):self.exp.evaluate(MODEL_NAMES[0])

    def test_original_architecture_checkpoint_compatibility(self):
        nb=json.loads((ROOT/"notebooks/milestone01_team01_dario.ipynb").read_text(encoding="utf-8"))
        from torch import nn
        from torchvision import models
        context={"nn":nn,"models":models,"IDS":IDS}
        exec("".join(nb["cells"][7]["source"]),context)
        for number,name in [(1,MODEL_NAMES[0]),(2,MODEL_NAMES[1])]:
            old=context["make_model"](number)
            new=make_model(name,False)
            new.load_state_dict(old.state_dict())
            self.assertEqual(list(old.state_dict()),list(new.state_dict()))
if __name__=="__main__":unittest.main()
