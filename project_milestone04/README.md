# Milestone 4 - End-to-end detection and identification

One command takes an image and returns every face it finds, with a bounding box and a name: one of our five identities, or "unknown" when the evidence is too weak. It combines the [Milestone 1 ResNet18 classifier](../project/models/group1_repro_v1/resnet18_frozen_backbone.pt) with the [Milestone 3 YOLOv8n detector](../project_milestone03/models/best.pt). Neither model was retrained.

| Deliverable | Where |
| --- | --- |
| Pipeline (library and command line) | [`pipeline.py`](pipeline.py) |
| Live demo (Gradio) | [`demo.py`](demo.py) |
| Validation selection and test scores | [`evaluate.py`](evaluate.py), results in [`results/summary.md`](results/summary.md) |
| Final results gallery | [`gallery/README.md`](gallery/README.md): 20 correct single faces, 7 errors with analysis, 10 multi-face scenes |
| Regression checks | [`test_pipeline.py`](test_pipeline.py) |

## How it works

```text
image ──► YOLOv8n on the image  ─┐
      └─► YOLOv8n on a ½-size    ├─► merge (NMS), drop boxes nested in a larger box ──► face boxes
          copy on a plain canvas ─┘                                                   + 5 detector scores
                                                                                            │
          each box ──► portrait crop (CelebA framing) ──► ResNet18 ──► 5 class probabilities │
                                                                                            ▼
          fused = 0.5 × detector scores (normalized) + 0.5 × ResNet18 probabilities
          name = top identity if fused confidence ≥ 0.61, otherwise "unknown" (the box stays)

no face found at either scale ──► ResNet18 classifies the whole image (no box)
```

- **Detector scores.** `model.predict()` only keeps each box's top class score, but the vote needs all five. `CelebrityPipeline.raw_detect()` runs Ultralytics' own preprocessing and NMS and reads the five scores of each kept box. Its boxes and confidences match `model.predict(..., agnostic_nms=True)` exactly; `test_pipeline.py` checks this. NMS is class-agnostic because the name is decided after detection, so each face should keep exactly one box.
- **Two scales.** The detector learned faces mostly 80-105 px wide in 800 × 600 scenes (middle half of the training boxes). On a close-up portrait it finds no face, or boxes an eye or a lock of hair (it found a face in only 1 of the 17 test portraits). A copy shrunk onto a canvas twice the size brings the face back to a familiar scale. Both scales are always searched and merged, so group photos and close-ups use the same path.
- **Portrait crop.** ResNet18 was trained on whole 178 × 218 CelebA photos, not tight face boxes. Each box is widened to the same framing: the photo is 2.00 × the face width and 1.87 × its height, with the face 0.51 and 0.57 box-widths from the left and top edges. These are the medians over all 121 source photos, measured with YuNet (`results/portrait_framing.json`).
- **Vote and rejection.** In a first check on the validation scenes (original scale only), YOLOv8n named 98 of 108 detected faces correctly. ResNet18 agreed with it on 83 of those 98, and on none of the 10 it got wrong. Disagreement lowers the fused confidence, so a threshold withholds most wrong names. The Milestone 3 detector had no way to say "none of the five", which the [TA image check](../project_milestone03/results/ta_image_check/README.md) flagged as the main gap.

## Setup

Use the Milestone 3 environment, or create it. Python 3.12 or 3.13, from the main repo folder.

macOS / Linux:

```sh
python3.13 -m venv .venv-m3
source .venv-m3/bin/activate
python -m pip install torch==2.14.0 torchvision==0.29.0
python -m pip install -r project_milestone04/requirements.txt
```

Windows (PowerShell):

```powershell
python -m venv .venv-m3
.venv-m3\Scripts\Activate.ps1
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu torch==2.14.0+cpu torchvision==0.29.0+cpu
python -m pip install -r project_milestone04/requirements.txt
```

No GPU is needed. One image takes well under a second on a laptop CPU, after a few seconds of model loading.

## Run

Live demo, at <http://127.0.0.1:7860>. Upload a photo, take a webcam snapshot, paste from the clipboard or click an example. Each face gets a coloured box with its name and fused confidence; grey boxes are "unknown". The table shows what each model said on its own. The sliders change the unknown threshold and the vote weight live. Add `--share` for a temporary public link.

```sh
python project_milestone04/demo.py
```

Several images or folders at once, without a browser. Each image prints its faces and gets an annotated copy in `tmp/m4_pipeline/`, plus a `predictions.json`:

```sh
python project_milestone04/pipeline.py photo1.jpg photo2.png some_folder/
```

From Python:

```python
import sys; sys.path.insert(0, "project_milestone04")
from pipeline import CelebrityPipeline, draw
pipeline = CelebrityPipeline()
faces, analysis = pipeline("project_milestone02/dataset/test/images/test_0009.jpg")
for face in faces:
    print(face["identity"], round(face["confidence"], 2), face["box"])
```

## Reproduce the reported numbers

```sh
python project_milestone04/evaluate.py --output tmp/m4_reproduce    # about 1 minute on CPU
python project_milestone04/gallery.py                               # rebuilds gallery/ from results/
python -m unittest discover -s project_milestone04 -p "test_*.py"
```

`evaluate.py` chooses every setting on validation, writes [`selected_config.json`](results/selected_config.json), then scores the test split once. Without `--output` it replaces `results/`. On macOS ARM CPU, repeat runs give identical files. Other platforms may differ in the last decimal of a confidence score.

## Results (held-out test)

Two test sets that no setting was chosen on: the 36 Milestone 2 test scenes (113 faces) and the 17 test source photos as single-face portraits. A face counts as correct if it gets the right name and a box with IoU ≥ 0.5. Precision is correct names divided by all names given. "Unknown" faces are boxed but not named.

| Method | Scenes: correct / wrong / unknown / missed | Scene accuracy | Scene precision | Portraits: correct / wrong / unknown | Portrait accuracy |
| --- | --- | ---: | ---: | --- | ---: |
| YOLOv8n alone (Milestone 3) | 104 / 8 / 0 / 1 (+1 extra box) | 0.920 | 0.920 | 0 / 1 / 0, 16 missed | 0.000 |
| ResNet18 alone (Milestone 1) | 71 / 41 / 0 / 1 | 0.628 | 0.634 | 11 / 6 / 0 | 0.647 |
| Pipeline, no rejection | 104 / 8 / 0 / 1 | 0.920 | 0.929 | 16 / 1 / 0 | 0.941 |
| **Pipeline (final)** | **85 / 1 / 26 / 1** | **0.752** | **0.988** | **16 / 0 / 1** | **0.941** |

ResNet18 alone on scenes classifies the pipeline's face crops. On portraits it classifies the whole photo, as in Milestone 1.

What this shows:

- **Single faces now work.** The detector alone missed 16 of 17 close-up portraits. The two-scale search finds all 17, and the pipeline names 16 correctly. That is 94%, against 65% for ResNet18 on the whole photo.
- **Wrong names almost disappear, at a price.** With rejection, wrong names in scenes drop from 8 to 1 and precision rises to 0.988. But 19 correctly named faces are withheld too, so accuracy falls from 0.920 to 0.752. Nearly all withheld faces belong to 7007 and 7: the pair the detector confuses, and where ResNet18 is weakest (33% on 7007 in Milestone 1). The [threshold curve](results/threshold_curve.png) shows the trade-off; the demo slider moves along it.
- **The vote does not raise top-1 accuracy.** Without rejection, the pipeline names the same 104 scene faces as YOLOv8n alone. ResNet18 adds a second opinion that exposes doubtful names; it does not fix them.
- **Strangers are only partly caught.** On the TA's real-photo grid, the pipeline withholds a name for 2 of the 6 people outside our five identities; Milestone 3 named all six. The other 4 still get a celebrity name: for 3 of them both models agree. Closed-set classifiers cannot reliably say "none of the five". See [`results/ta_check.csv`](results/ta_check.csv) and [`ta_check.jpg`](results/ta_check.jpg).

Per identity, with the test scenes and portraits together ([`test_per_identity.csv`](results/test_per_identity.csv)):

| Identity | Faces | Correct | Wrong | Unknown | Missed | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 7007 | 27 | 13 | 1 | 12 | 1 | 0.481 |
| 2970 | 27 | 27 | 0 | 0 | 0 | 1.000 |
| 2336 | 28 | 28 | 0 | 0 | 0 | 1.000 |
| 7 | 26 | 13 | 0 | 13 | 0 | 0.500 |
| 4428 | 22 | 20 | 0 | 2 | 0 | 0.909 |
| **All** | **130** | **101** | **1** | **27** | **1** | **0.777** |

## Settings chosen on validation

Validation has 36 scenes (111 faces) and 19 single-face portraits. Each candidate is scored as correct names minus wrong names, so withholding a name only helps when it prevents an error. Full grids are in [`validation_detection.csv`](results/validation_detection.csv) and [`validation_grid.csv`](results/validation_grid.csv).

| Setting | Candidates | Selected |
| --- | --- | --- |
| Detection | original image, canvas only when nothing is found / both scales always | both scales |
| Canvas factor | 2, 3, 4 | 2 |
| YOLOv8n weight in the vote | 0.0 to 1.0 in steps of 0.1 | 0.5 |
| Unknown threshold | none, or 0.30 to 0.90 in steps of 0.01 | 0.61 |
| Detector confidence, NMS IoU | fixed from the Milestone 3 sweep | 0.50, 0.70 (class-agnostic) |

Several candidates score within one or two faces of each other on validation (net 106-109). The selection is therefore a reasonable choice, not a precise optimum.

**How the settings changed.** The first version searched the canvas only when the original image gave no box. Its test result was 84 correct, 0 wrong and 28 unknown in scenes, and 15 / 1 / 1 on portraits. The gallery then showed the cause of most portrait errors: the original-scale pass had boxed an eye or a lock of hair, so the canvas search never ran. We then added the always-on two-scale option and the nested-box rule, and re-selected every setting on validation. Test results were looked at between the two versions, so the final test numbers are slightly less independent than a single blind run. The report should say so.

## Limitations

- **Five identities, closed set.** Unknown-person rejection relies on two models disagreeing. When both are confidently wrong, as for 4 of the 6 strangers in the TA grid, the stranger is named.
- **ResNet18 is the weak partner.** It saw only 17 training photos per person and reaches 63-65% alone. A stronger classifier, or a face-embedding model with a distance threshold, would make the vote and the rejection more useful.
- **Synthetic scenes.** All scene metrics come from pasted portraits on drawn backgrounds. Real group photos have occlusion, profile views and lighting the detector never saw.
- **Small test sets.** 113 scene faces and 17 portraits. One face is almost 1 percentage point of scene accuracy and 6 points of portrait accuracy.

## Files

| File or folder | Purpose |
| --- | --- |
| `pipeline.py` | `CelebrityPipeline` (detect, crop, classify, vote), drawing, and the command line for files and folders |
| `demo.py` | Gradio app for the live demonstration |
| `evaluate.py` | Validation selection, test scoring, baselines, TA check and `results/summary.md` |
| `gallery.py` | Builds `gallery/` from `results/predictions.csv` |
| `test_pipeline.py` | Detector equivalence, crop geometry, fusion, nested boxes and scoring rules |
| `results/` | Selected settings, validation grids, test metrics, per-face predictions, threshold curve, TA check |
| `gallery/` | Final results gallery with contact sheets for the report |
