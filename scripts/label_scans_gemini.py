"""Recognize numbered ten-glyph cards with Gemini 3.5 Flash or Flash-Lite."""

import argparse
import base64
import json
import os
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.scans.extract import read_manifest, write_manifest


MODELS = ("gemini-3.5-flash", "gemini-3.5-flash-lite")


def api_key() -> str:
    value = os.environ.get("GEMINI_API_KEY", "").strip()
    if value:
        return value
    env_file = Path(__file__).resolve().parents[1] / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            if "=" not in line or line.lstrip().startswith("#"):
                continue
            name, raw = line.split("=", 1)
            if name.strip() == "GEMINI_API_KEY":
                return raw.strip().strip('"\'').strip()
    raise RuntimeError("GEMINI_API_KEY is absent from the environment and project .env")


def make_card(records: list[dict[str, str]], path: Path) -> None:
    """Show raw and cleaned crops side by side so recognition can check artifacts."""
    tile_w, tile_h, columns = 280, 170, 5
    rows = (len(records) + columns - 1) // columns
    card = Image.new("RGB", (tile_w * columns, tile_h * rows), "white")
    draw = ImageDraw.Draw(card)
    for i, record in enumerate(records):
        x, y = (i % columns) * tile_w, (i // columns) * tile_h
        draw.rectangle((x + 2, y + 2, x + tile_w - 2, y + tile_h - 2), outline="#888888", width=2)
        draw.text((x + 10, y + 8), f"{i + 1:02d}   original            cleaned", fill="black")
        for offset, key in ((8, "raw_path"), (145, "clean_path")):
            with Image.open(record[key]) as original:
                glyph = original.convert("RGB")
                glyph.thumbnail((122, 122), Image.Resampling.LANCZOS)
                card.paste(glyph, (x + offset + (122 - glyph.width) // 2, y + 35 + (122 - glyph.height) // 2))
    path.parent.mkdir(parents=True, exist_ok=True)
    card.save(path)
    path.with_suffix(".json").write_text(json.dumps({str(i + 1): r["id"] for i, r in enumerate(records)}, ensure_ascii=False, indent=2), encoding="utf-8")


def recognize_card(path: Path, count: int, model: str, key: str) -> dict[int, tuple[str, int]]:
    prompt = (
        "Each numbered tile is ONE independent handwritten Traditional Chinese glyph. "
        "The left image is the original scan; the right image has its background removed. "
        "Compare both images because cleaning can remove faint strokes. Read only the visible strokes. "
        "Do not infer from adjacent tiles, sentence meaning, common words, or expected grammar. "
        "Preserve Traditional forms; do not silently convert to Simplified Chinese. "
        f"Return exactly one result for every index from 1 through {count}. "
        "For each tile return char (one visible Han character) and confidence (integer 1-10). "
        "Confidence is subjective certainty from the strokes, NOT a calibrated probability: "
        "10 means unmistakable, 7-9 strong, 4-6 plausible but ambiguous, 1-3 weak visual guess. "
        "If a plausible single character exists despite ambiguity, give the best visual guess with a low score. "
        "For a blank, punctuation-only, or genuinely unreadable tile, return char='' and confidence=1. "
        "Never fabricate a character when no plausible strokes support it."
    )
    schema = {"type": "OBJECT", "properties": {"results": {"type": "ARRAY", "items": {
        "type": "OBJECT", "properties": {"index": {"type": "INTEGER"}, "char": {"type": "STRING"},
                                      "confidence": {"type": "INTEGER", "minimum": 1, "maximum": 10}},
        "required": ["index", "char", "confidence"]}}}, "required": ["results"]}
    payload = {"contents": [{"role": "user", "parts": [
        {"text": prompt},
        {"inline_data": {"mime_type": "image/png", "data": base64.b64encode(path.read_bytes()).decode("ascii")}},
    ]}], "generationConfig": {"responseMimeType": "application/json", "responseSchema": schema}}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    request = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={
        "Content-Type": "application/json", "x-goog-api-key": key,
    }, method="POST")
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                data = json.load(response)
            parts = data["candidates"][0]["content"]["parts"]
            body = "".join(part.get("text", "") for part in parts)
            parsed = json.loads(body)
            results = {}
            for item in parsed["results"]:
                index = item.get("index")
                value = unicodedata.normalize("NFC", str(item.get("char", "")).strip())
                confidence = item.get("confidence")
                if type(confidence) is not int or not 1 <= confidence <= 10:
                    raise ValueError(f"Invalid Gemini confidence for tile {index}: {confidence!r}")
                if isinstance(index, int) and 1 <= index <= count and index not in results:
                    valid = len(value) == 1 and not value.isspace()
                    results[index] = (value if valid else "", confidence if valid else 1)
            if len(results) != count:
                raise ValueError(f"Gemini returned {len(results)} of {count} tile indices")
            return results
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == 3:
                raise RuntimeError(f"Gemini HTTP {exc.code}; response body withheld to protect credentials") from None
            time.sleep(2 ** attempt)
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == 3:
                raise RuntimeError(f"Gemini request failed: {type(exc).__name__}") from None
            time.sleep(2 ** attempt)
    raise RuntimeError("Gemini retry limit reached")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--model", choices=MODELS, default="gemini-3.5-flash")
    parser.add_argument("--card-size", type=int, default=10)
    parser.add_argument("--limit", type=int, help="Maximum number of candidate cells to process")
    parser.add_argument("--cards-only", action="store_true", help="Generate cards without API calls")
    parser.add_argument("--auto-accept", action="store_true", help="Use valid single-character answers in training without manual review")
    parser.add_argument("--refresh-confidence", action="store_true", help="Requery previous Gemini cells missing confidence; preserve human decisions")
    args = parser.parse_args()
    if not 1 <= args.card_size <= 10:
        parser.error("--card-size must be between 1 and 10")
    records = read_manifest(args.manifest)
    pending = [r for r in records if r["status"] == "unlabeled" or
               (args.refresh_confidence and r["gemini_model"] and not r["gemini_confidence"] and r["status"] != "blank")]
    if args.limit is not None:
        pending = pending[:args.limit]
    if not pending:
        print("No unlabeled cells to process")
        return
    key = None if args.cards_only else api_key()
    card_dir = args.manifest.parent / "cards"
    for start in range(0, len(pending), args.card_size):
        batch = pending[start:start + args.card_size]
        card = card_dir / f"card_{batch[0]['id']}.png"
        make_card(batch, card)
        if args.cards_only:
            print(f"Card: {card}")
            continue
        answers = recognize_card(card, len(batch), args.model, key)
        for index, row in enumerate(batch, 1):
            value, confidence = answers[index]
            human_reviewed = row["label_source"] == "human_review"
            row["gemini_label"] = value
            row["gemini_model"] = args.model
            row["gemini_confidence"] = str(confidence)
            if not human_reviewed:
                row["status"] = ("accepted" if args.auto_accept else "proposed") if value else "rejected"
                row["label"] = value
                row["label_source"] = ("gemini_auto" if args.auto_accept else "gemini_proposal") if value else ""
        write_manifest(args.manifest, records)
        print(f"Card {start // args.card_size + 1}: {sum(bool(answers[i][0]) for i in answers)} proposed, {sum(not answers[i][0] for i in answers)} rejected")
    if args.cards_only:
        print("No API requests made")


if __name__ == "__main__":
    main()
