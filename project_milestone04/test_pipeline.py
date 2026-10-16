"""Regression checks for the Milestone 4 pipeline: detector equivalence, crop geometry, fusion and scoring.

    python -m unittest discover -s project_milestone04 -p "test_*.py"
"""

import json
import unittest

import numpy as np

import evaluate
import pipeline as m4


def face(identity, box, confidence=0.9):
    return {"identity": identity, "box": list(box), "confidence": confidence}


class GeometryAndFusion(unittest.TestCase):
    def test_portrait_crop_inverts_aligned_face_box(self):
        # Cropping around the expected face box of an aligned 178x218 photo should give back the whole photo.
        photo = np.random.default_rng(0).integers(0, 255, (218, 178, 3), dtype=np.uint8)
        crop = m4.portrait_crop(photo, evaluate.aligned_face_box(218, 178))
        self.assertEqual(crop.shape, (218, 178, 3))
        self.assertLess(np.abs(crop.astype(int) - photo.astype(int)).mean(), 2)

    def test_crop_near_border_is_padded_not_cut(self):
        crop = m4.portrait_crop(np.zeros((100, 100, 3), np.uint8), [0, 0, 40, 50])
        self.assertEqual(crop.shape, (218, 178, 3))

    def test_fusion_and_unknown_threshold(self):
        detector, classifier = np.array([0.9, 0.1, 0, 0, 0]), np.array([0.2, 0.6, 0.1, 0.05, 0.05])
        fused = m4.fuse(detector, classifier, 0.5)
        self.assertAlmostEqual(fused.sum(), 1.0, places=6)
        self.assertEqual(int(fused.argmax()), 0)
        raw = {"box": [0, 0, 1, 1], "detector_scores": detector, "classifier_probs": classifier}
        self.assertEqual(m4.decide_face(raw, "detected", 0.5, 0.0)["identity"], m4.IDS[0])
        self.assertEqual(m4.decide_face(raw, "detected", 0.5, 0.99)["identity"], m4.UNKNOWN)

    def test_whole_image_fallback_uses_classifier_only(self):
        classifier = np.array([0.1, 0.1, 0.7, 0.05, 0.05])
        self.assertTrue(np.allclose(m4.fuse(None, classifier, 0.9), classifier))

    def test_nested_box_is_dropped(self):
        boxes = np.array([[10, 10, 110, 130], [40, 50, 60, 70], [200, 10, 300, 130]], float)
        kept, scores = m4.drop_nested(boxes, np.eye(3)[:, :3])
        self.assertEqual(kept.tolist(), [[10, 10, 110, 130], [200, 10, 300, 130]])
        self.assertEqual(len(scores), 2)


class Scoring(unittest.TestCase):
    truth = [("7007", [0, 0, 10, 10])]

    def test_correct_name_is_matched_before_a_duplicate(self):
        rows = evaluate.score_scene(self.truth, [face("7", [0, 0, 10, 10.5]), face("7007", [0, 0, 10, 11])])
        self.assertEqual(sorted(row["outcome"] for row in rows), ["correct", "extra_named"])

    def test_unknown_is_neither_correct_nor_wrong(self):
        summary = evaluate.summarize(evaluate.score_scene(self.truth, [face(m4.UNKNOWN, [0, 0, 10, 10])]))
        self.assertEqual((summary["correct"], summary["wrong"], summary["unknown"]), (0, 0, 1))
        self.assertEqual(summary["precision"], 0.0)

    def test_low_overlap_is_a_miss_plus_extra_box(self):
        rows = evaluate.score_scene(self.truth, [face("7007", [6, 6, 16, 16])])
        self.assertEqual(sorted(row["outcome"] for row in rows), ["extra_named", "missed"])

    def test_settings_were_selected_on_validation(self):
        config = json.loads(m4.CONFIG.read_text())
        self.assertEqual(config["selection_split"], "val")
        self.assertIn(config["detection"], m4.DETECTION_MODES)
        self.assertTrue(set(m4.DEFAULTS) <= set(config))


class Models(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = m4.CelebrityPipeline()

    def test_raw_detection_matches_ultralytics_predict(self):
        for name in ("val_0000.jpg", "val_0007.jpg", "val_0021.jpg"):
            path = evaluate.DATASET / "val" / "images" / name
            boxes, scores = self.pipeline.raw_detect(np.ascontiguousarray(m4.load_rgb(path)[..., ::-1]))
            result = self.pipeline.detector.predict(str(path), conf=m4.CONFIDENCE, iou=m4.NMS_IOU, agnostic_nms=True,
                                                    imgsz=m4.IMAGE_SIZE, device="cpu", verbose=False)[0]
            order = np.argsort(boxes[:, 0])
            expected = result.boxes.xyxy.numpy()[np.argsort(result.boxes.xyxy.numpy()[:, 0])]
            self.assertTrue(np.allclose(boxes[order], expected, atol=0.5), name)
            self.assertTrue(np.allclose(np.sort(scores.max(1)), np.sort(result.boxes.conf.numpy()), atol=1e-4), name)

    def test_close_up_portrait_is_found_and_named(self):
        faces, analysis = self.pipeline(m4.ROOT / "data" / "identities" / "2970" / "112624.jpg")
        self.assertNotEqual(analysis["stage"], "whole_image")
        self.assertEqual([f["identity"] for f in faces], ["2970"])

    def test_scene_returns_one_box_per_face(self):
        faces, _ = self.pipeline(evaluate.DATASET / "test" / "images" / "test_0008.jpg")
        truth = (evaluate.DATASET / "test" / "labels" / "test_0008.txt").read_text().split("\n")
        self.assertEqual(len(faces), len([line for line in truth if line.strip()]))


if __name__ == "__main__":
    unittest.main()
