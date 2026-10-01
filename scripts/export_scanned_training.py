"""Validate reviewed scan cells and export a training manifest."""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.scans.extract import read_manifest


def is_han(char: str) -> bool:
    if len(char) != 1:
        return False
    code = ord(char)
    return any(lo <= code <= hi for lo, hi in ((0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF), (0x20000, 0x2FA1F)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/processed_scans/train_manifest.csv"))
    parser.add_argument("--include-proposed", action="store_true", help="Also use unreviewed Gemini single-character proposals")
    args = parser.parse_args()
    rows = read_manifest(args.manifest)
    selected = []
    for row in rows:
        if row["status"] != "accepted" and not (args.include_proposed and row["status"] == "proposed"):
            continue
        if not is_han(row["label"]):
            continue
        if not Path(row["clean_path"]).is_file():
            raise FileNotFoundError(row["clean_path"])
        selected.append({key: row[key] for key in (
            "id", "clean_path", "label", "writer_id", "source_file", "page", "row", "col", "label_source",
            "gemini_label", "gemini_confidence", "gemini_model", "ocr_label", "ocr_confidence", "ocr_engine")})
    if not selected:
        raise RuntimeError("No accepted Han characters; review labels or use --auto-accept during Gemini labeling")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    print(f"Training pairs: {len(selected)} cells, {len({r['label'] for r in selected})} unique Han characters -> {args.output}")


if __name__ == "__main__":
    main()
