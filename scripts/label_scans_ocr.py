"""Run a local OCR baseline on each extracted scan cell without changing review decisions."""

import argparse
import importlib.metadata
import math
import os
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.export_scanned_training import is_han
from src.scans.extract import read_manifest, write_manifest


def recognize_variant(engine, path: str) -> tuple[str, float]:
    result = engine(path, use_det=False, use_cls=False, use_rec=True)
    texts = getattr(result, "txts", None) or ()
    scores = getattr(result, "scores", None) or ()
    text = unicodedata.normalize("NFC", str(texts[0]).strip()) if texts else ""
    score = float(scores[0]) if scores else 0.0
    if not math.isfinite(score):
        score = 0.0
    return text, max(0.0, min(1.0, score))


def choose_character(raw: tuple[str, float], clean: tuple[str, float]) -> tuple[str, str, str]:
    candidates = [(text, score, variant) for variant, (text, score) in (("raw", raw), ("clean", clean)) if is_han(text)]
    if not candidates:
        return "", "", ""
    text, score, variant = max(candidates, key=lambda item: item[1])
    return text, f"{score:.4f}", variant


def process_rows(rows: list[dict[str, str]], engine, engine_name: str, *, limit: int | None = None,
                 overwrite: bool = False, include_blank: bool = False, save=None) -> int:
    processed = 0
    for row in rows:
        if row["status"] == "blank" and not include_blank:
            continue
        if row["ocr_engine"] and not overwrite:
            continue
        raw = recognize_variant(engine, row["raw_path"])
        clean = recognize_variant(engine, row["clean_path"])
        label, confidence, image = choose_character(raw, clean)
        row.update(ocr_label=label, ocr_confidence=confidence, ocr_engine=engine_name, ocr_image=image,
                   ocr_raw_text=raw[0], ocr_raw_confidence=f"{raw[1]:.4f}",
                   ocr_clean_text=clean[0], ocr_clean_confidence=f"{clean[1]:.4f}")
        processed += 1
        if save:
            save(rows)
        if limit is not None and processed >= limit:
            break
    return processed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--limit", type=int, help="Process at most this many eligible cells")
    parser.add_argument("--overwrite", action="store_true", help="Recompute cells already scored by OCR")
    parser.add_argument("--include-blank", action="store_true", help="Also inspect cells marked blank by extraction")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    try:
        from rapidocr import RapidOCR
    except ImportError as exc:
        raise RuntimeError("OCR requires rapidocr and onnxruntime; install requirements-ocr.txt") from exc
    engine = RapidOCR(params={"Global.log_level": "error"})
    engine_name = f"rapidocr-{importlib.metadata.version('rapidocr')}/PP-OCRv6"
    rows = read_manifest(args.manifest)

    def save(updated):
        temporary = args.manifest.with_suffix(".ocr.tmp")
        write_manifest(temporary, updated)
        os.replace(temporary, args.manifest)

    processed = process_rows(rows, engine, engine_name, limit=args.limit, overwrite=args.overwrite,
                             include_blank=args.include_blank, save=save)
    print(f"OCR processed {processed} cells; {sum(bool(r['ocr_label']) for r in rows)} have one Han candidate")


if __name__ == "__main__":
    main()
