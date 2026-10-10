"""Live demo: run the fine-tuned YOLOv8 celebrity detector on an image or webcam frame.

Owner: Murat Akca. Loads the committed checkpoint and uses the settings selected on validation
by the hyperparameter sweep (confidence 0.50, NMS IoU 0.70, class-aware NMS).

    python project_milestone03/demo.py                                  # Gradio app (upload or webcam)
    python project_milestone03/demo.py --image path/to/photo.jpg        # command line, saves annotated copy
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "project_milestone03" / "models" / "best.pt"
TEST_IMAGES = ROOT / "project_milestone02" / "dataset" / "test" / "images"
IDENTITIES = ["7007", "2970", "2336", "7", "4428"]
# From results/dario_sweep_*/selected_config.json; chosen on validation before the test split was scored.
CONFIDENCE, NMS_IOU, IMAGE_SIZE = 0.50, 0.70, 640
# Gallery examples: three clean successes, two instructive failures and the TA's real-photo grid.
EXAMPLES = [TEST_IMAGES / name for name in ["test_0005.jpg", "test_0008.jpg", "test_0013.jpg", "test_0002.jpg", "test_0009.jpg"]]
EXAMPLES.append(ROOT / "project_milestone03" / "results" / "ta_image_check" / "ta_test_image.png")


def load_model(checkpoint=CHECKPOINT):
    from ultralytics import YOLO

    if not Path(checkpoint).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}. See project_milestone03/README.md.")
    model = YOLO(str(checkpoint))
    names = [str(model.names[index]) for index in range(len(model.names))]
    if model.task != "detect" or names != IDENTITIES:
        raise ValueError(f"Expected the five-identity detection checkpoint; found task={model.task}, names={names}.")
    return model


def detect(model, image, confidence=CONFIDENCE, nms_iou=NMS_IOU, agnostic_nms=False, device="cpu"):
    """Return (annotated BGR image, detection rows). `image` is a path or a BGR array."""
    result = model.predict(image, conf=confidence, iou=nms_iou, agnostic_nms=agnostic_nms,
                           imgsz=IMAGE_SIZE, device=device, verbose=False)[0]
    rows = []
    for box, score, label in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist()):
        rows.append([model.names[int(label)], round(score, 3), *[round(value) for value in box]])
    rows.sort(key=lambda row: row[2])  # left to right
    return result.plot(), rows


def run_cli(model, image_path, output, args):
    import cv2

    annotated, rows = detect(model, str(image_path), args.confidence, args.nms_iou, args.agnostic_nms, args.device)
    # Default to the git-ignored tmp/ folder so the dataset folders stay untouched.
    output = Path(output or ROOT / "tmp" / "demo" / f"{Path(image_path).stem}_detections.jpg")
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), annotated)
    print(f"{len(rows)} face(s) detected in {image_path}")
    for identity, score, x1, y1, x2, y2 in rows:
        print(f"  identity {identity:>4}  confidence {score:.3f}  box ({x1}, {y1}, {x2}, {y2})")
    print(f"Annotated image saved to {output}")


def run_app(model, args):
    import gradio as gr

    def predict(image, confidence, nms_iou, agnostic_nms):
        if image is None:
            return None, []
        # Gradio supplies RGB; Ultralytics expects BGR arrays and plots in BGR.
        annotated, rows = detect(model, image[..., ::-1], confidence, nms_iou, agnostic_nms, args.device)
        return annotated[..., ::-1], rows

    examples = [[str(path), CONFIDENCE, NMS_IOU, False] for path in EXAMPLES if path.is_file()]
    with gr.Blocks(title="IE 7615 Group 1 - YOLOv8 celebrity detection") as app:
        gr.Markdown("## IE 7615 Group 1 — YOLOv8 celebrity detection\n"
                    f"Fine-tuned YOLOv8n, five identities ({', '.join(IDENTITIES)}). Upload a photo, use the webcam "
                    "or pick an example below (five held-out test scenes, then the TA's real-photo grid). The model can only answer with these five IDs, so any "
                    "other face is either missed or assigned one of them.")
        with gr.Row():
            with gr.Column():
                image = gr.Image(label="Input", type="numpy", sources=["upload", "webcam", "clipboard"])
                confidence = gr.Slider(0.05, 0.95, value=CONFIDENCE, step=0.05, label="Confidence threshold")
                nms_iou = gr.Slider(0.10, 0.90, value=NMS_IOU, step=0.05, label="NMS IoU threshold")
                agnostic_nms = gr.Checkbox(value=False, label="Class-agnostic NMS (one ID per overlapping face)")
                button = gr.Button("Detect", variant="primary")
            with gr.Column():
                output = gr.Image(label="Detections")
                table = gr.Dataframe(headers=["identity", "confidence", "x1", "y1", "x2", "y2"], label="Boxes")
        inputs = [image, confidence, nms_iou, agnostic_nms]
        button.click(predict, inputs, [output, table])
        image.change(predict, inputs, [output, table])
        if examples:
            gr.Examples(examples, inputs, [output, table], fn=predict, run_on_click=True)
    app.launch(server_name=args.host, server_port=args.port, share=args.share)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", type=Path, help="Run once on this image instead of starting the app.")
    parser.add_argument("--output", type=Path, help="Where --image saves the annotated copy.")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--confidence", type=float, default=CONFIDENCE)
    parser.add_argument("--nms-iou", type=float, default=NMS_IOU)
    parser.add_argument("--agnostic-nms", action="store_true")
    parser.add_argument("--device", default="cpu", help="cpu, mps or a CUDA index such as 0.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create a temporary public Gradio link (useful on Colab).")
    args = parser.parse_args()
    model = load_model(args.checkpoint)
    if args.image:
        run_cli(model, args.image, args.output, args)
    else:
        run_app(model, args)


if __name__ == "__main__":
    main()
