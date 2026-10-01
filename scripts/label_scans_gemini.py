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
from io import BytesIO
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


def recognize_card(path: Path, count: int, model: str, key: str) -> dict[int, str]:
    prompt = (
        "Read each numbered tile as one isolated handwritten Traditional Chinese character. "
        "Each tile has an original crop on the left and a cleaned version on the right. "
        "Use the visible strokes only. Do not correct grammar, infer from neighboring tiles, "
        "or replace Traditional Chinese with Simplified Chinese. "
        f"Return exactly one result for each number 1 through {count}. "
        "For an empty, punctuation-only, illegible, or uncertain tile, return an empty string. "
        "For readable tiles, return exactly one visible character."
    )
    schema = {"type": "OBJECT", "properties": {"results": {"type": "ARRAY", "items": {
        "type": "OBJECT", "properties": {"index": {"type": "INTEGER"}, "char": {"type": "STRING"}},
        "required": ["index", "char"]}}}, "required": ["results"]}
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
                if isinstance(index, int) and 1 <= index <= count and index not in results:
                    results[index] = value if len(value) == 1 and not value.isspace() else ""
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
    args = parser.parse_args()
    if not 1 <= args.card_size <= 10:
        parser.error("--card-size must be between 1 and 10")
    records = read_manifest(args.manifest)
    pending = [r for r in records if r["status"] == "unlabeled"]
    if args.limit is not None:
        pending = pending[:args.limit]
    if not pending:
        print("No unlabeled cells to process")
        return
    key = None if args.cards_only else api_key()
    card_dir = args.manifest.parent / "cards"
    for start in range(0, len(pending), args.card_size):
        batch = pending[start:start + args.card_size]
        card = card_dir / f"card_{start // args.card_size + 1:04d}.png"
        make_card(batch, card)
        if args.cards_only:
            print(f"Card: {card}")
            continue
        answers = recognize_card(card, len(batch), args.model, key)
        for index, row in enumerate(batch, 1):
            value = answers[index]
            row["gemini_label"] = value
            row["gemini_model"] = args.model
            row["status"] = ("accepted" if args.auto_accept else "proposed") if value else "rejected"
            if value:
                row["label"] = value
                row["label_source"] = "gemini_auto" if args.auto_accept else "gemini_proposal"
        write_manifest(args.manifest, records)
        print(f"Card {start // args.card_size + 1}: {sum(bool(answers[i]) for i in answers)} proposed, {sum(not answers[i] for i in answers)} rejected")
    if args.cards_only:
        print("No API requests made")


if __name__ == "__main__":
    main()
