"""Suggest and validate a ruled page's outer grid before manual adjustment."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image

from src.scans.extract import _runs, green_mask


def _centers(runs: list[tuple[int, int]]) -> list[float]:
    return [(start + end) / 2 for start, end in runs]


def suggest_grid(image: Image.Image) -> dict:
    """Return an editable first guess, including row/column counts when possible."""
    mask = green_mask(np.asarray(image.convert("RGB")))
    height, width = mask.shape
    horizontal_strength = mask.sum(axis=1)
    horizontal_threshold = max(width * 0.08, float(horizontal_strength.max()) * 0.34)
    horizontal = _centers(_runs(np.flatnonzero(horizontal_strength > horizontal_threshold)))
    if len(horizontal) >= 2:
        top, bottom = horizontal[0], horizontal[-1]
        row_pitch = float(np.median(np.diff(horizontal)))
        rows = max(1, round((bottom - top) / row_pitch)) if row_pitch > 2 else 1
    else:
        top, bottom, rows = height * 0.08, height * 0.92, 22

    vertical_strength = mask[max(0, int(top)):min(height, int(bottom) + 1)].sum(axis=0)
    vertical_threshold = max((bottom - top) * 0.12, float(vertical_strength.max()) * 0.25)
    vertical = _centers(_runs(np.flatnonzero(vertical_strength > vertical_threshold)))
    if len(vertical) >= 2:
        left, right = vertical[0], vertical[-1]
        col_pitch = float(np.median(np.diff(vertical)))
        cols = max(1, round((right - left) / col_pitch)) if col_pitch > 2 else 1
    else:
        left, right, cols = width * 0.05, width * 0.95, 26

    corners = [[round(left, 1), round(top, 1)], [round(right, 1), round(top, 1)],
               [round(right, 1), round(bottom, 1)], [round(left, 1), round(bottom, 1)]]
    return {"corners": corners, "cols": cols, "rows": rows,
            "detected_horizontal_lines": len(horizontal), "detected_vertical_lines": len(vertical)}


def validate_calibration(value: dict, width: int, height: int) -> dict:
    try:
        cols, rows = int(value["cols"]), int(value["rows"])
        corners = [[float(x), float(y)] for x, y in value["corners"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Provide four corners and integer column/row counts") from exc
    if not (1 <= cols <= 100 and 1 <= rows <= 100):
        raise ValueError("Column and row counts must be between 1 and 100")
    if len(corners) != 4 or any(not math.isfinite(n) for pair in corners for n in pair):
        raise ValueError("Exactly four finite corner coordinates are required")
    if any(not (0 <= x < width and 0 <= y < height) for x, y in corners):
        raise ValueError("Corners must lie inside the preview image")
    turns = []
    for i in range(4):
        a, b, c = corners[i], corners[(i + 1) % 4], corners[(i + 2) % 4]
        turns.append((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
    if min(turns) <= 100:
        raise ValueError("Corners must form a non-crossing quadrilateral in UL, UR, LR, LL order")
    return {"cols": cols, "rows": rows, "corners": corners}
