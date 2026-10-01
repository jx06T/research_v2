"""Review proposed scan labels, optionally applying corrections from a CSV file."""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.scans.extract import read_manifest, write_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--export", type=Path, help="Write review sheet with id, candidate, final_label, decision")
    parser.add_argument("--apply", type=Path, help="Apply edited review sheet")
    args = parser.parse_args()
    rows = read_manifest(args.manifest)
    if args.export:
        args.export.parent.mkdir(parents=True, exist_ok=True)
        with args.export.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=["id", "raw_path", "clean_path", "candidate", "final_label", "decision"])
            writer.writeheader()
            for row in rows:
                if row["status"] in ("proposed", "rejected", "unlabeled"):
                    writer.writerow({"id": row["id"], "raw_path": row["raw_path"], "clean_path": row["clean_path"],
                                     "candidate": row["gemini_label"], "final_label": row["label"], "decision": ""})
        print(f"Review sheet: {args.export}")
    if args.apply:
        by_id = {row["id"]: row for row in rows}
        changed = 0
        with args.apply.open("r", newline="", encoding="utf-8-sig") as stream:
            for review in csv.DictReader(stream):
                decision = (review.get("decision") or "").strip().lower()
                if not decision:
                    continue
                if decision not in ("accept", "reject"):
                    raise ValueError(f"Invalid decision for {review.get('id')}: {decision}")
                row = by_id.get(review.get("id"))
                if row is None:
                    raise ValueError(f"Unknown ID: {review.get('id')}")
                label = (review.get("final_label") or "").strip()
                if decision == "accept" and len(label) != 1:
                    raise ValueError(f"Accepted label must contain exactly one character: {row['id']}")
                row["status"] = "accepted" if decision == "accept" else "rejected"
                row["label"] = label if decision == "accept" else ""
                row["label_source"] = "human_review" if decision == "accept" else ""
                changed += 1
        write_manifest(args.manifest, rows)
        print(f"Applied {changed} review decisions")


if __name__ == "__main__":
    main()
