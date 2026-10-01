"""Extract one glyph candidate per cell from a ruled, vertical-writing page."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


FIELDS = [
    "id", "source_file", "page", "writer_id", "row", "col", "reading_index",
    "raw_path", "clean_path", "bbox", "ink_fraction", "status", "label",
    "label_source", "gemini_label", "gemini_model", "review_note",
]


def load_pages(path: Path):
    if path.suffix.lower() == ".pdf":
        try:
            import pypdfium2 as pdfium
        except ImportError as exc:
            raise RuntimeError("PDF input requires pypdfium2 (pip install -r requirements.txt)") from exc
        pdf = pdfium.PdfDocument(str(path))
        try:
            for number in range(len(pdf)):
                page = pdf[number]
                try:
                    yield number + 1, page.render(scale=2).to_pil().convert("RGB")
                finally:
                    page.close()
        finally:
            pdf.close()
    else:
        with Image.open(path) as image:
            yield 1, image.convert("RGB")


def green_mask(rgb: np.ndarray) -> np.ndarray:
    a = rgb.astype(np.int16)
    return (a[:, :, 1] > a[:, :, 0] + 7) & (a[:, :, 1] > a[:, :, 2] + 2) & (a[:, :, 0] < 245) & (a[:, :, 1] > 90)


def _runs(indices: np.ndarray) -> list[tuple[int, int]]:
    if len(indices) == 0:
        return []
    breaks = np.where(np.diff(indices) > 1)[0] + 1
    return [(int(part[0]), int(part[-1])) for part in np.split(indices, breaks)]


def auto_corners(image: Image.Image, cols: int, rows: int) -> list[tuple[float, float]]:
    """Find the outside grid intersections from long pale-green rules."""
    mask = green_mask(np.asarray(image))
    h, w = mask.shape
    ys = mask.sum(axis=1)
    y_candidates = _runs(np.flatnonzero(ys > max(w * 0.24, float(ys.max()) * 0.42)))
    if len(y_candidates) < rows + 1:
        raise ValueError("Could not locate all horizontal rules; provide --corners")
    y0 = (y_candidates[0][0] + y_candidates[0][1]) / 2
    y1 = (y_candidates[-1][0] + y_candidates[-1][1]) / 2
    near = mask[max(0, int(y0)):min(h, int(y1) + 1)]
    xs = near.sum(axis=0)
    x_candidates = _runs(np.flatnonzero(xs > max((y1 - y0) * 0.30, float(xs.max()) * 0.30)))
    centers = [(a + b) / 2 for a, b in x_candidates]
    strengths = [float(xs[a:b + 1].max()) for a, b in x_candidates]
    if len(centers) < cols:
        raise ValueError("Could not locate enough vertical rules; provide --corners")
    pitch = float(np.median(np.diff(centers)))
    # A watermark can create an extra line near a real rule; keep the stronger peak.
    kept = []
    for center, strength in zip(centers, strengths):
        if kept and center - kept[-1][0] < pitch * 0.55:
            if strength > kept[-1][1]:
                kept[-1] = (center, strength)
        else:
            kept.append((center, strength))
    centers = [p[0] for p in kept]
    if len(centers) == cols:
        # An outer vertical rule may be missing while horizontal rules still show its extent.
        horizontal_extents = []
        for a, b in y_candidates:
            band = mask[max(0, a - 1):min(h, b + 2)].sum(axis=0)
            active = np.flatnonzero(band > 0)
            if len(active):
                horizontal_extents.append((int(active.min()), int(active.max())))
        left_edge = float(np.median([p[0] for p in horizontal_extents]))
        right_edge = float(np.median([p[1] for p in horizontal_extents]))
        if centers[0] - left_edge > right_edge - centers[-1]:
            centers.insert(0, centers[0] - pitch)
        else:
            centers.append(centers[-1] + pitch)
    if len(centers) != cols + 1:
        raise ValueError("Ambiguous grid columns; provide --corners")
    x0, x1 = centers[0], centers[-1]
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def rectify(image: Image.Image, corners: list[tuple[float, float]]) -> Image.Image:
    if len(corners) != 4:
        raise ValueError("Four corners required: top-left, top-right, bottom-right, bottom-left")
    ul, ur, lr, ll = [np.asarray(p, dtype=float) for p in corners]
    width = max(1, round((np.linalg.norm(ur - ul) + np.linalg.norm(lr - ll)) / 2))
    height = max(1, round((np.linalg.norm(ll - ul) + np.linalg.norm(lr - ur)) / 2))
    dest = [(0, 0), (width, 0), (width, height), (0, height)]
    equations, targets = [], []
    for (u, v), (x, y) in zip(dest, corners):
        equations.extend([[u, v, 1, 0, 0, 0, -x * u, -x * v],
                          [0, 0, 0, u, v, 1, -y * u, -y * v]])
        targets.extend([x, y])
    coeff = np.linalg.solve(np.asarray(equations, dtype=float), np.asarray(targets, dtype=float))
    return image.transform((width, height), Image.Transform.PERSPECTIVE,
                           tuple(coeff), resample=Image.Resampling.BICUBIC)


def _line_positions(signal: np.ndarray, count: int) -> list[int]:
    """Refine an evenly spaced lattice at each expected rule position."""
    maximum = len(signal) - 1
    step = maximum / count
    result = []
    for index in range(count + 1):
        center = index * step
        radius = max(2, int(step * 0.22))
        lo = max(0, round(center - radius))
        hi = min(maximum, round(center + radius))
        position = lo + int(np.argmax(signal[lo:hi + 1]))
        result.append(position)
    # Faint or obscured rules may peak at the same pixel; the regular lattice is a safe fallback.
    if any(b - a < step * 0.55 for a, b in zip(result, result[1:])):
        return [round(i * step) for i in range(count + 1)]
    result[0], result[-1] = 0, maximum
    return result


def clean_cell(raw: Image.Image, size: int = 128) -> tuple[Image.Image, float]:
    """Suppress colored paper marks; retain dark handwriting as grayscale."""
    a = np.asarray(raw.convert("RGB")).astype(np.float32)
    gray = a @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    ink = np.clip((210.0 - gray) / 170.0, 0, 1)
    green = green_mask(a.astype(np.uint8))
    # Keep dark handwriting even where it crosses a green rule.
    ink[green & (gray > 105)] = 0
    ink[ink < 0.12] = 0
    fraction = float(np.mean(ink > 0.22))
    coords = np.argwhere(ink > 0.22)
    canvas = Image.new("L", (size, size), 255)
    if len(coords):
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        glyph = Image.fromarray(np.uint8(np.clip(255 * (1 - ink[y0:y1, x0:x1]), 0, 255)))
        scale = min(size * 0.8 / glyph.width, size * 0.8 / glyph.height)
        new_size = (max(1, round(glyph.width * scale)), max(1, round(glyph.height * scale)))
        glyph = glyph.resize(new_size, Image.Resampling.LANCZOS)
        canvas.paste(glyph, ((size - glyph.width) // 2, (size - glyph.height) // 2))
    return canvas, fraction


def extract_page(image: Image.Image, source: Path, page: int, output: Path,
                 corners: list[tuple[float, float]], cols: int, rows: int,
                 writer_id: str, size: int = 128) -> list[dict[str, str]]:
    output.mkdir(parents=True, exist_ok=True)
    rect = rectify(image, corners)
    rect.save(output / "rectified.png")
    mask = green_mask(np.asarray(rect))
    x_lines = _line_positions(mask.sum(axis=0), cols)
    y_lines = _line_positions(mask.sum(axis=1), rows)
    overlay = rect.copy()
    draw = ImageDraw.Draw(overlay)
    for x in x_lines:
        draw.line([(x, 0), (x, rect.height)], fill=(255, 0, 0), width=2)
    for y in y_lines:
        draw.line([(0, y), (rect.width, y)], fill=(255, 0, 0), width=2)
    overlay.save(output / "grid_overlay.png")
    (output / "raw").mkdir(exist_ok=True)
    (output / "clean").mkdir(exist_ok=True)
    source_id = hashlib.sha256(f"{source.resolve()}:{page}".encode()).hexdigest()[:10]
    records = []
    # Physical columns increase left-to-right; reading order is right-to-left.
    for col in range(cols - 1, -1, -1):
        for row in range(rows):
            index = (cols - 1 - col) * rows + row
            cell_id = f"{source.stem}_p{page:03d}_{source_id}_c{col + 1:02d}_r{row + 1:02d}"
            x0, x1 = x_lines[col], x_lines[col + 1]
            y0, y1 = y_lines[row], y_lines[row + 1]
            inset = max(1, round(min(x1 - x0, y1 - y0) * 0.025))
            box = (x0 + inset, y0 + inset, x1 - inset, y1 - inset)
            raw = rect.crop(box)
            clean, fraction = clean_cell(raw, size=size)
            raw_path = output / "raw" / f"{cell_id}.png"
            clean_path = output / "clean" / f"{cell_id}.png"
            raw.save(raw_path)
            clean.save(clean_path)
            records.append({
                "id": cell_id, "source_file": str(source.resolve()), "page": str(page),
                "writer_id": writer_id, "row": str(row + 1), "col": str(col + 1),
                "reading_index": str(index), "raw_path": str(raw_path.resolve()),
                "clean_path": str(clean_path.resolve()), "bbox": json.dumps(box),
                "ink_fraction": f"{fraction:.5f}",
                "status": "blank" if fraction < 0.003 else "unlabeled",
                "label": "", "label_source": "", "gemini_label": "", "gemini_model": "", "review_note": "",
            })
    return records


def write_manifest(path: Path, records: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))
