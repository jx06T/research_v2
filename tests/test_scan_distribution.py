import csv
import json
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.package_scan_dataset import pack, unpack
from scripts.review_scans_web import make_server
from scripts.scan_workflow import reset_lite_proposals
from src.scans.extract import FIELDS, read_manifest, write_manifest


class DistributionTests(unittest.TestCase):
    def test_package_round_trip_without_png_in_git(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            glyph = root / "glyph.png"
            pixels = np.full((128, 128), 255, dtype=np.uint8)
            pixels[40:80, 50:70] = 30
            Image.fromarray(pixels).save(glyph)
            manifest = root / "train_manifest.csv"
            with manifest.open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream, fieldnames=["id", "clean_path", "label", "writer_id", "source_file", "page", "row", "col", "label_source"])
                writer.writeheader()
                writer.writerow({"id": "one", "clean_path": str(glyph), "label": "同", "writer_id": "writer_a", "source_file": "private.pdf", "page": "1", "row": "1", "col": "1", "label_source": "human_review"})
            package = root / "portable.npz"
            pack(manifest, package)
            self.assertTrue(package.with_suffix(".json").is_file())
            self.assertEqual(json.loads(package.with_name("portable_labels.jsonl").read_text(encoding="utf-8"))["label"], "同")
            destination = unpack(package, root / "cloud")
            with destination.open(encoding="utf-8-sig", newline="") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual((row["label"], row["writer_id"]), ("同", "writer_a"))
            with Image.open(row["clean_path"]) as image:
                np.testing.assert_array_equal(np.asarray(image), pixels)

    def test_review_api_persists_label_and_audit_event(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            page = root / "page_001"
            (page / "raw").mkdir(parents=True)
            (page / "clean").mkdir()
            for path in (page / "raw" / "one.png", page / "clean" / "one.png", page / "rectified.png"):
                Image.new("L", (20, 20), 255).save(path)
            row = dict.fromkeys(FIELDS, "")
            row.update(id="one", raw_path=str(page / "raw" / "one.png"), clean_path=str(page / "clean" / "one.png"),
                       bbox="[0,0,20,20]", status="proposed", gemini_label="司", gemini_model="gemini-3.5-flash")
            manifest = root / "cells.csv"
            write_manifest(manifest, [row])
            server = make_server(manifest, "127.0.0.1", 0)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                with urllib.request.urlopen(base + "/api/rows") as response:
                    self.assertEqual(json.load(response)[0]["status"], "proposed")
                request = urllib.request.Request(base + "/api/review", data=json.dumps({"id": "one", "decision": "accept", "label": "同", "note": "checked"}).encode(), headers={"Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(request) as response:
                    self.assertEqual(json.load(response)["label"], "同")
                self.assertEqual(read_manifest(manifest)[0]["label_source"], "human_review")
                reject = urllib.request.Request(base + "/api/review", data=json.dumps({"id": "one", "decision": "reject", "label": ""}).encode(), headers={"Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(reject) as response:
                    self.assertEqual(json.load(response)["label_source"], "human_review")
                self.assertEqual(read_manifest(manifest)[0]["status"], "rejected")
                self.assertEqual(len((root / "review_events.jsonl").read_text(encoding="utf-8").splitlines()), 2)
                with urllib.request.urlopen(base + "/image/one/clean") as response:
                    self.assertEqual(response.headers.get_content_type(), "image/png")
            finally:
                server.shutdown()
                thread.join(timeout=2)
                server.server_close()

    def test_old_audited_rejection_is_restored_as_reviewed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "cells.csv"
            row = dict.fromkeys(FIELDS, "")
            row.update(id="one", status="rejected", label_source="")
            write_manifest(manifest, [row])
            (root / "review_events.jsonl").write_text(json.dumps({"id": "one", "after": {"status": "rejected"}}) + "\n", encoding="utf-8")
            server = make_server(manifest, "127.0.0.1", 0)
            try:
                self.assertEqual(read_manifest(manifest)[0]["label_source"], "human_review")
            finally:
                server.server_close()

    def test_relabel_lite_preserves_human_reviews(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest = Path(temp) / "cells.csv"
            rows = []
            for number, status in enumerate(("proposed", "accepted"), 1):
                row = dict.fromkeys(FIELDS, "")
                row.update(id=str(number), status=status, label="同", gemini_label="同",
                           gemini_model="gemini-3.5-flash-lite", label_source="human_review" if status == "accepted" else "gemini_proposal")
                rows.append(row)
            write_manifest(manifest, rows)
            self.assertEqual(reset_lite_proposals(manifest), 1)
            updated = read_manifest(manifest)
            self.assertEqual(updated[0]["status"], "unlabeled")
            self.assertEqual(updated[1]["status"], "accepted")
            self.assertEqual(len(list(manifest.parent.glob("cells_before_flash_relabel_*.csv"))), 1)


if __name__ == "__main__":
    unittest.main()
