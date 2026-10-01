import json
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw

from scripts.calibrate_scans_web import make_server
from src.scans.calibration import suggest_grid, validate_calibration


class ScanCalibrationTests(unittest.TestCase):
    def test_grid_guess_and_corner_validation(self):
        image = Image.new("RGB", (600, 400), "white")
        draw = ImageDraw.Draw(image)
        for x in range(50, 451, 100):
            draw.line((x, 60, x, 300), fill=(155, 205, 175), width=2)
        for y in range(60, 301, 80):
            draw.line((50, y, 450, y), fill=(155, 205, 175), width=2)
        guess = suggest_grid(image)
        self.assertEqual((guess["cols"], guess["rows"]), (4, 3))
        self.assertLess(abs(guess["corners"][0][0] - 50), 3)
        self.assertEqual(validate_calibration(guess, 600, 400)["cols"], 4)
        with self.assertRaises(ValueError):
            validate_calibration({"cols": 4, "rows": 3, "corners": [guess["corners"][0], guess["corners"][2],
                                                                     guess["corners"][1], guess["corners"][3]]}, 600, 400)

    def test_web_apply_saves_configuration_without_extracting_until_wrapper_runs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output = root / "source.png", root / "prepared"
            Image.new("RGB", (200, 150), "white").save(source)
            server = make_server(source, output, "127.0.0.1", 0)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                with urllib.request.urlopen(base + "/api/config") as response:
                    self.assertEqual(json.load(response)["width"], 200)
                payload = {"cols": 4, "rows": 3, "corners": [[10, 10], [190, 10], [190, 140], [10, 140]]}
                request = urllib.request.Request(base + "/api/apply", data=json.dumps(payload).encode(),
                                                 headers={"Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(request) as response:
                    self.assertTrue(json.load(response)["saved"])
                self.assertTrue(server.applied)
                self.assertEqual(json.loads((output / "manual_calibration.json").read_text(encoding="utf-8"))["rows"], 3)
                self.assertFalse((output / "cells.csv").exists())
            finally:
                server.shutdown()
                thread.join(timeout=2)
                server.server_close()


if __name__ == "__main__":
    unittest.main()
