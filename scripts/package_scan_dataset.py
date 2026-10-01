"""Pack reviewed glyph pixels and labels into a portable, Git-friendly NPZ file."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


FORMAT_VERSION = 1
PROVENANCE_FIELDS = ("gemini_label", "gemini_confidence", "gemini_model", "ocr_label", "ocr_confidence", "ocr_engine")


def pack(manifest: Path, output: Path):
    with manifest.open("r", newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Training manifest is empty")
    images, labels, ids, writers, sources = [], [], [], [], []
    provenance = {key: [] for key in PROVENANCE_FIELDS}
    shape = None
    for row in rows:
        with Image.open(row["clean_path"]) as image:
            pixels = np.asarray(image.convert("L"), dtype=np.uint8)
        if shape is None:
            shape = pixels.shape
        if pixels.shape != shape or shape[0] != shape[1]:
            raise ValueError(f"Inconsistent glyph dimensions: {row['id']}")
        if len(row["label"]) != 1:
            raise ValueError(f"Expected one-character label: {row['id']}")
        images.append(pixels)
        labels.append(row["label"])
        ids.append(row["id"])
        writers.append(row["writer_id"])
        sources.append(row.get("label_source", ""))
        for key in PROVENANCE_FIELDS:
            provenance[key].append(row.get(key, ""))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, images=np.stack(images), labels=np.asarray(labels),
                        ids=np.asarray(ids), writer_ids=np.asarray(writers),
                        label_sources=np.asarray(sources), format_version=np.asarray(FORMAT_VERSION),
                        **{key: np.asarray(values) for key, values in provenance.items()})
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    metadata = {"format_version": FORMAT_VERSION, "count": len(labels), "unique_characters": len(set(labels)),
                "size": shape[0], "writers": sorted(set(writers)), "sha256": digest,
                "contains_unreviewed": "gemini_proposal" in sources}
    output.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    labels_path = output.with_name(output.stem + "_labels.jsonl")
    with labels_path.open("w", encoding="utf-8") as stream:
        for i, (cell_id, label, writer, source) in enumerate(zip(ids, labels, writers, sources)):
            record = {"id": cell_id, "label": label, "writer_id": writer, "label_source": source}
            record.update({key: values[i] for key, values in provenance.items()})
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Package: {output} ({len(labels)} glyphs, {output.stat().st_size / 1048576:.2f} MiB)")
    return metadata


def unpack(package: Path, output: Path):
    metadata_path = package.with_suffix(".json")
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if hashlib.sha256(package.read_bytes()).hexdigest() != metadata["sha256"]:
            raise ValueError("Package checksum mismatch")
    with np.load(package, allow_pickle=False) as data:
        if int(data["format_version"]) != FORMAT_VERSION:
            raise ValueError("Unsupported scan package version")
        images = data["images"]
        labels = data["labels"]
        ids = data["ids"]
        writers = data["writer_ids"]
        sources = data["label_sources"]
        provenance = {key: data[key] if key in data.files else [""] * len(labels) for key in PROVENANCE_FIELDS}
        if images.ndim != 3 or images.shape[0] != len(labels):
            raise ValueError("Invalid scan package dimensions")
        glyph_dir = output / "glyphs"
        glyph_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for i in range(len(labels)):
            image_path = glyph_dir / f"glyph_{i:06d}.png"
            Image.fromarray(images[i], mode="L").save(image_path)
            row = {"id": str(ids[i]), "clean_path": str(image_path.resolve()),
                         "label": str(labels[i]), "writer_id": str(writers[i]),
                         "source_file": "", "page": "", "row": "", "col": "",
                         "label_source": str(sources[i])}
            row.update({key: str(values[i]) for key, values in provenance.items()})
            rows.append(row)
    manifest = output / "train_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Unpacked {len(rows)} glyphs -> {manifest}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p_pack = sub.add_parser("pack")
    p_pack.add_argument("manifest", type=Path)
    p_pack.add_argument("--output", type=Path, required=True)
    p_unpack = sub.add_parser("unpack")
    p_unpack.add_argument("package", type=Path)
    p_unpack.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "pack":
        pack(args.manifest, args.output)
    else:
        unpack(args.package, args.output)


if __name__ == "__main__":
    main()
