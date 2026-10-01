"""Compare OCR and Gemini confidence with human-accepted scan labels."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.export_scanned_training import is_han
from src.scans.extract import read_manifest


def bucket(channel: str, score: float) -> str:
    if channel == "Gemini":
        return "1-3" if score <= 3 else "4-6" if score <= 6 else "7-8" if score <= 8 else "9-10"
    return "0-.49" if score < .5 else ".50-.79" if score < .8 else ".80-.89" if score < .9 else ".90-1"


def report(rows: list[dict[str, str]]) -> dict:
    gold = [r for r in rows if r["status"] == "accepted" and r["label_source"] == "human_review" and is_han(r["label"])]
    channels = {}
    for name, label_key, score_key, ran_key in (("Gemini", "gemini_label", "gemini_confidence", "gemini_model"),
                                                ("OCR", "ocr_label", "ocr_confidence", "ocr_engine")):
        scored = [r for r in gold if r[ran_key] and (r[score_key] or name == "OCR")]
        bins = {}
        for row in scored:
            group = bucket(name, float(row[score_key] or 0))
            entry = bins.setdefault(group, {"count": 0, "correct": 0})
            entry["count"] += 1
            entry["correct"] += row[label_key] == row["label"]
        channels[name] = {"evaluated": len(scored), "with_candidate": sum(bool(r[label_key]) for r in scored),
                          "correct": sum(r[label_key] == r["label"] for r in scored),
                          "bins": bins}
    both = [r for r in gold if r["gemini_label"] and r["ocr_label"]]
    return {"human_accepted_han": len(gold), "channels": channels,
            "both_have_candidate": len(both),
            "channels_disagree": sum(r["gemini_label"] != r["ocr_label"] for r in both)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    result = report(read_manifest(args.manifest))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("Only human-accepted Han cells are compared; this selected subset is not an unbiased page-level accuracy estimate.")


if __name__ == "__main__":
    main()
