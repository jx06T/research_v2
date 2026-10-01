import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from src.scans.extract import FIELDS, auto_corners, clean_cell, extract_page, read_manifest, write_manifest
from scripts.label_scans_gemini import recognize_card
from scripts.label_scans_ocr import process_rows
from scripts.export_scanned_training import is_han


class ScanPipelineTests(unittest.TestCase):
    def test_grid_order_and_cleaning(self):
        image = Image.new("RGB", (600, 400), "white")
        draw = ImageDraw.Draw(image)
        for x in range(50, 451, 100):
            draw.line((x, 60, x, 300), fill=(155, 205, 175), width=2)
        for y in range(60, 301, 80):
            draw.line((50, y, 450, y), fill=(155, 205, 175), width=2)
        draw.line((375, 90, 420, 120), fill="black", width=5)
        corners = auto_corners(image, cols=4, rows=3)
        for actual, expected in zip(corners, [(50, 60), (450, 60), (450, 300), (50, 300)]):
            self.assertLess(abs(actual[0] - expected[0]), 3)
            self.assertLess(abs(actual[1] - expected[1]), 3)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            rows = extract_page(image, output / "source.png", 1, output, corners, 4, 3, "writer_a")
            self.assertEqual(len(rows), 12)
            self.assertEqual((rows[0]["col"], rows[0]["row"]), ("4", "1"))
            self.assertEqual(rows[0]["status"], "unlabeled")
            self.assertEqual(rows[1]["status"], "blank")
            write_manifest(output / "cells.csv", rows)
            self.assertEqual(len(read_manifest(output / "cells.csv")), 12)

    def test_background_color_removal(self):
        image = Image.new("RGB", (80, 80), "white")
        draw = ImageDraw.Draw(image)
        draw.line((0, 10, 79, 10), fill=(155, 205, 175), width=2)
        draw.line((35, 25, 45, 65), fill="black", width=3)
        clean, fraction = clean_cell(image)
        self.assertGreater(fraction, 0.005)
        self.assertLess(int(np.asarray(clean).min()), 100)

    def test_gemini_schema_and_single_character_validation(self):
        class FakeResponse:
            def __enter__(self):
                return io.BytesIO(json.dumps({"candidates": [{"content": {"parts": [{"text": json.dumps({
                    "results": [{"index": 1, "char": "同", "confidence": 8},
                                {"index": 2, "char": "猜測", "confidence": 6},
                                {"index": 3, "char": "", "confidence": 1}]
                }, ensure_ascii=False)}]}}]}).encode())

            def __exit__(self, *_):
                return False

        with tempfile.TemporaryDirectory() as folder:
            card = Path(folder) / "card.png"
            Image.new("RGB", (10, 10), "white").save(card)
            with patch("urllib.request.urlopen", return_value=FakeResponse()) as call:
                answers = recognize_card(card, 3, "gemini-3.5-flash", "test-key")
            self.assertEqual(answers, {1: ("同", 8), 2: ("", 1), 3: ("", 1)})
            request = call.call_args.args[0]
            self.assertIn("gemini-3.5-flash", request.full_url)
            self.assertEqual(request.get_header("X-goog-api-key"), "test-key")
            payload = json.loads(request.data)
            self.assertIn("confidence", payload["generationConfig"]["responseSchema"]["properties"]["results"]["items"]["required"])

    def test_ocr_scores_both_crops_without_changing_human_label(self):
        class FakeResult:
            def __init__(self, text, score):
                self.txts, self.scores = (text,), (score,)

        class FakeEngine:
            def __call__(self, path, **options):
                self_options = (options["use_det"], options["use_cls"], options["use_rec"])
                self_test.assertEqual(self_options, (False, False, True))
                return FakeResult("同", 0.44) if path == "raw.png" else FakeResult("同", 0.82)

        self_test = self
        row = dict.fromkeys(FIELDS, "")
        row.update(id="one", raw_path="raw.png", clean_path="clean.png", status="accepted",
                   label="問", label_source="human_review")
        self.assertEqual(process_rows([row], FakeEngine(), "rapidocr-test"), 1)
        self.assertEqual((row["label"], row["label_source"]), ("問", "human_review"))
        self.assertEqual((row["ocr_label"], row["ocr_confidence"], row["ocr_image"]), ("同", "0.8200", "clean"))

    def test_training_labels_are_han(self):
        self.assertTrue(is_han("同"))
        self.assertFalse(is_han("。"))
        self.assertFalse(is_han("AB"))

    def test_old_manifest_loads_with_empty_confidence_fields(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest = Path(folder) / "cells.csv"
            manifest.write_text("id,status,gemini_label\none,proposed,同\n", encoding="utf-8")
            row = read_manifest(manifest)[0]
            self.assertEqual((row["gemini_label"], row["gemini_confidence"], row["ocr_label"]), ("同", "", ""))
            write_manifest(manifest, [row])
            self.assertEqual(read_manifest(manifest)[0]["gemini_label"], "同")


if __name__ == "__main__":
    unittest.main()
