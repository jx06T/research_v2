"""Prepare ruled scan pages for glyph recognition."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.scans.extract import auto_corners, extract_page, load_pages, write_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="A PDF, image, or directory")
    parser.add_argument("--output", type=Path, default=Path("data/processed_scans"))
    parser.add_argument("--cols", type=int, default=26)
    parser.add_argument("--rows", type=int, default=22)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--writer-id", default="unknown")
    parser.add_argument("--corners", type=float, nargs=8, metavar=("UL_X", "UL_Y", "UR_X", "UR_Y", "LR_X", "LR_Y", "LL_X", "LL_Y"))
    args = parser.parse_args()
    if args.cols < 1 or args.rows < 1 or args.size < 16:
        parser.error("cols, rows and size must be positive")
    paths = sorted(p for p in args.input.rglob("*") if p.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}) if args.input.is_dir() else [args.input]
    if not paths or any(not p.is_file() for p in paths):
        parser.error("No input files found")
    if args.corners and len(paths) != 1:
        parser.error("--corners requires one input file")
    records = []
    for path in paths:
        for page, image in load_pages(path):
            corners = list(zip(args.corners[::2], args.corners[1::2])) if args.corners else auto_corners(image, args.cols, args.rows)
            page_dir = args.output / path.stem / f"page_{page:03d}"
            page_dir.mkdir(parents=True, exist_ok=True)
            (page_dir / "calibration.json").write_text(json.dumps({"source": str(path.resolve()), "page": page, "corners": corners, "cols": args.cols, "rows": args.rows, "rendered_size": image.size}, ensure_ascii=False, indent=2), encoding="utf-8")
            page_records = extract_page(image, path, page, page_dir, corners, args.cols, args.rows, args.writer_id, args.size)
            records.extend(page_records)
            print(f"{path.name} page {page}: {len(page_records)} cells, overlay: {page_dir / 'grid_overlay.png'}")
    manifest = args.output / "cells.csv"
    write_manifest(manifest, records)
    print(f"Manifest: {manifest}; blank={sum(r['status'] == 'blank' for r in records)}, candidates={sum(r['status'] == 'unlabeled' for r in records)}")


if __name__ == "__main__":
    main()
