"""Live demo of the Milestone 4 pipeline: YOLOv8n finds faces, ResNet18 and YOLOv8n vote on who they are.

Owner: Murat Akca. Uses the settings chosen on validation (results/selected_config.json).

    python project_milestone04/demo.py            # Gradio app: upload, webcam or paste an image
    python project_milestone04/demo.py --share    # temporary public link (for example on Colab)

For several images at once without a browser, use pipeline.py:

    python project_milestone04/pipeline.py photo1.jpg photo2.png folder/
"""

from __future__ import annotations

import argparse

import pipeline as m4

TEST_SCENES = m4.ROOT / "project_milestone02" / "dataset" / "test" / "images"
EXAMPLES = [TEST_SCENES / "test_0008.jpg", TEST_SCENES / "test_0009.jpg",
            m4.ROOT / "data" / "identities" / "2970" / "112624.jpg",   # held-out test portrait
            m4.ROOT / "data" / "identities" / "7" / "090940.jpg",      # held-out test portrait
            m4.ROOT / "project_milestone03" / "results" / "ta_image_check" / "ta_test_image.png"]
HEADERS = ["identity", "fused confidence", "YOLOv8n", "YOLOv8n score", "ResNet18", "ResNet18 probability",
           "box (x1, y1, x2, y2)", "detection stage"]


def table(faces):
    rows = []
    for face in faces:
        box = "whole image" if face["box"] is None else ", ".join(f"{v:.0f}" for v in face["box"])
        score = "" if face["detector_score"] is None else round(face["detector_score"], 3)
        rows.append([face["identity"], round(face["confidence"], 3), face["detector_identity"] or "-", score,
                     face["classifier_identity"], round(face["classifier_probability"], 3), box, face["stage"]])
    return rows


def run_app(pipeline, args):
    import gradio as gr

    settings = pipeline.settings

    def predict(image, unknown_threshold, yolo_weight):
        if image is None:
            return None, [], ""
        faces, analysis = pipeline(image, yolo_weight, unknown_threshold)
        named = sum(face["identity"] != m4.UNKNOWN for face in faces)
        summary = (f"**{len(faces)} face(s)**, {named} named, {len(faces) - named} withheld as unknown. "
                   f"Detection stage: `{analysis['stage']}`.")
        return m4.draw(image, faces), table(faces), summary

    examples = [[str(path), settings["unknown_threshold"], settings["yolo_weight"]] for path in EXAMPLES
                if path.is_file()]
    with gr.Blocks(title="IE 7615 Group 1 - celebrity detection and identification") as app:
        gr.Markdown(
            "## IE 7615 Group 1 — celebrity detection and identification\n"
            f"YOLOv8n (Milestone 3) finds each face; ResNet18 (Milestone 1) classifies a portrait crop of it; the two "
            f"identity scores are averaged. Five identities: {', '.join(m4.IDS)}. A face whose fused confidence is "
            f"below the threshold is boxed in grey as **unknown** instead of being given a name. If no face is "
            f"found, ResNet18 classifies the whole image.")
        with gr.Row():
            with gr.Column():
                image = gr.Image(label="Input", type="numpy", sources=["upload", "webcam", "clipboard"])
                threshold = gr.Slider(0.0, 0.95, value=settings["unknown_threshold"], step=0.01,
                                      label=f"Unknown threshold (validated: {settings['unknown_threshold']})")
                weight = gr.Slider(0.0, 1.0, value=settings["yolo_weight"], step=0.1,
                                   label=f"YOLOv8n weight in the vote (validated: {settings['yolo_weight']})")
                button = gr.Button("Identify", variant="primary")
            with gr.Column():
                output = gr.Image(label="Pipeline output")
                summary = gr.Markdown()
        faces = gr.Dataframe(headers=HEADERS, label="Faces")
        inputs, outputs = [image, threshold, weight], [output, faces, summary]
        button.click(predict, inputs, outputs)
        image.change(predict, inputs, outputs)
        threshold.release(predict, inputs, outputs)
        weight.release(predict, inputs, outputs)
        if examples:
            gr.Examples(examples, inputs, outputs, fn=predict, run_on_click=True,
                        label="Examples: two test scenes, two test portraits, the TA's real-photo grid")
    app.launch(server_name=args.host, server_port=args.port, share=args.share)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cpu", help="cpu, mps or a CUDA index such as 0.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create a temporary public Gradio link.")
    args = parser.parse_args()
    run_app(m4.CelebrityPipeline(device=args.device), args)


if __name__ == "__main__":
    main()
