# Best-model handoff

**Selected model:** frozen-backbone ResNet18, epoch 28, from `group1_repro_v1`. Validation accuracy is 80.0%; held-out test accuracy is 66.7% (10/15).

## Saved artifact

- Checkpoint: `models/group1_repro_v1/resnet18_frozen_backbone.pt`
- SHA-256: `e3a8199480649cf8a63b64c0be0491bd4efa45d3d46fa6bab63757b63d8693b3`
- Split: `configs/splits/group1_repro_v1_seed47.csv`
- Split SHA-256: `a4c1ecb50ce5a24e63f85296759c9d7d5de4e065de81971d4bfe949520d8e0b7`
- Class order: **7007, 2970, 2336, 7, 4428** (output indices 0–4).

All files are included. Run `python scripts/verify_saved_run.py` from the repository root to verify the artifact and reproduce all test predictions without retraining.

## Load for inference

```python
import torch
from PIL import Image
from torchvision import transforms
from src.pipeline import make_model, IDS

saved = torch.load(
    "models/group1_repro_v1/resnet18_frozen_backbone.pt",
    map_location="cpu", weights_only=True,
)
assert saved["ids"] == IDS
model = make_model("resnet18_frozen_backbone", pretrained=False)
model.load_state_dict(saved["state_dict"])
model.eval()
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
with Image.open("data/identities/7/002065.jpg") as image:
    x = preprocess(image.convert("RGB")).unsqueeze(0)
with torch.no_grad():
    predicted_identity = IDS[model(x).argmax(1).item()]
print(predicted_identity)
```

Loading uses `pretrained=False` because the checkpoint already contains every trained weight; no ImageNet download is needed. ResNet18 has 11,179,077 total parameters and 2,565 trainable head parameters.

## Later milestones

Milestone 2 creates synthetic images and YOLO-format annotations. Milestone 3 trains YOLOv8. Keep this classifier as the Milestone 1 baseline and classification component for the broader system; these ResNet weights cannot be loaded into YOLOv8.

The current classifier expects a single centered face and one of the five known identities. Applying it to detector crops changes the input distribution and needs separate evaluation. It has no trained unknown-identity rejection class.
