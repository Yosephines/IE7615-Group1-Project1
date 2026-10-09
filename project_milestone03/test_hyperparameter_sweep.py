"""Regression checks for threshold metrics, data integrity and validation-only selection."""

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import hyperparameter_sweep as sweep


def detection(class_id=0, box=(0.1, 0.1, 0.3, 0.3), confidence=0.9):
    return {"class_id": class_id, "box": list(box), "confidence": confidence}


class SweepChecks(unittest.TestCase):
    def test_iou_geometry(self):
        self.assertEqual(sweep.box_iou((0, 0, 1, 1), (0, 0, 1, 1)), 1)
        self.assertEqual(sweep.box_iou((0, 0, 1, 1), (1, 1, 2, 2)), 0)
        self.assertAlmostEqual(sweep.box_iou((0, 0, 2, 2), (1, 1, 3, 3)), 1 / 7)

    def test_duplicate_cannot_match_one_face_twice(self):
        matches, used_truth, used_predictions = sweep.match_detections([detection()], [detection(), detection()])
        self.assertEqual(len(matches), 1)
        self.assertEqual(len(used_truth), 1)
        self.assertEqual(len(used_predictions), 1)

    def test_identity_mismatch_is_not_a_true_positive(self):
        matches, _, _ = sweep.match_detections([detection()], [detection(class_id=1)])
        self.assertEqual(matches, [])

    def test_confidence_sweep_counts_duplicates_and_misses(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "val" / "images").mkdir(parents=True)
            (root / "val" / "labels").mkdir()
            (root / "predictions").mkdir()
            (root / "val" / "images" / "example.jpg").touch()
            (root / "val" / "labels" / "example.txt").write_text("0 0.2 0.2 0.2 0.2\n1 0.6 0.6 0.2 0.2\n")
            (root / "predictions" / "example.txt").write_text(
                "0 0.2 0.2 0.2 0.2 0.9\n0 0.2 0.2 0.2 0.2 0.4\n2 0.6 0.6 0.2 0.2 0.6\n")
            low = sweep.evaluate_saved_predictions(root, root / "predictions", "val", 0.25)["overall"]
            high = sweep.evaluate_saved_predictions(root, root / "predictions", "val", 0.75)["overall"]
            self.assertEqual((low["tp"], low["fp"], low["fn"]), (1, 2, 1))
            self.assertEqual(low["identity_confusions"], 1)
            self.assertAlmostEqual(low["precision"], 1 / 3)
            self.assertEqual((high["tp"], high["fp"], high["fn"]), (1, 0, 1))
            self.assertEqual(high["mean_matched_iou"], 1)

    def test_missing_prediction_file_counts_as_no_detections(self):
        self.assertEqual(sweep.read_labels(Path("missing_prediction.txt"), predictions=True), [])
        with self.assertRaises(FileNotFoundError):
            sweep.read_labels(Path("missing_truth.txt"))

    def test_invalid_ground_truth_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            label = Path(temporary) / "invalid.txt"
            for value in ["0 0.1 0.1 0.6 0.6", "5 0.5 0.5 0.1 0.1", "0 nan 0.5 0.1 0.1"]:
                label.write_text(value)
                with self.assertRaises(ValueError):
                    sweep.read_labels(label)

    def test_zero_matches_have_no_mean_iou(self):
        scores = sweep.summarize_counts({"tp": 0, "fp": 0, "fn": 3, "ious": []})
        self.assertEqual(scores["f1"], 0)
        self.assertIsNone(scores["mean_matched_iou"])

    def test_default_grid_varies_both_required_parameters(self):
        grid = sweep.build_grid([0.1, 0.25, 0.5, 0.75], [0.3, 0.5, 0.7], [False, True])
        self.assertEqual(len(grid), 24)
        self.assertIn(sweep.BASELINE, grid)
        with self.assertRaises(ValueError):
            sweep.build_grid([float("nan")], [0.7], [False])

    def test_selection_refuses_test_scores(self):
        row = {"split": "test"}
        with self.assertRaises(ValueError):
            sweep.choose_configuration([row])

    def test_selection_uses_f1_before_ap(self):
        rows = []
        for confidence, f1, ap in [(0.25, 0.8, 0.95), (0.5, 0.9, 0.85)]:
            configuration = {"confidence": confidence, "nms_iou": 0.7, "agnostic_nms": False}
            rows.append({"split": "val", "config_id": sweep.config_id(configuration), **configuration,
                         "f1": f1, "precision": f1, "recall": f1, "map50_95": ap})
        self.assertEqual(sweep.choose_configuration(rows)["confidence"], 0.5)

    def test_archive_reads_only_unique_checkpoint_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "training.zip"
            with zipfile.ZipFile(archive, "w") as stream:
                stream.writestr("nested/weights/best.pt", b"checkpoint fixture")
                stream.writestr("../../outside.py", "raise RuntimeError('not executed')")
            path = sweep.resolve_checkpoint(None, archive, root)
            self.assertEqual(path.read_bytes(), b"checkpoint fixture")
            self.assertFalse((root.parent / "outside.py").exists())

    def test_supplied_dataset_matches_existing_evaluation(self):
        dataset = sweep.ROOT / "project_milestone02" / "dataset"
        audit = sweep.audit_dataset(dataset)
        reference = json.loads((sweep.ROOT / "project_milestone03" / "results" / "metrics.json").read_text())
        self.assertEqual(audit["dataset_sha256"], reference["dataset"]["dataset_sha256"])
        self.assertEqual(audit["shared_source_photos"], 0)
        self.assertEqual(audit["splits"]["train"]["images"], 504)
        self.assertEqual(audit["splits"]["val"]["faces"], 111)
        self.assertEqual(audit["splits"]["test"]["faces"], 113)


if __name__ == "__main__":
    unittest.main(verbosity=2)
