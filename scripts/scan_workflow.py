"""One entry point for scan extraction, Gemini/OCR labeling, review, packaging and training."""

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
import webbrowser
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
sys.path.insert(0, str(ROOT))


def run_script(name: str, *args: str, capture=False):
    command = [PYTHON, str(ROOT / "scripts" / name), *map(str, args)]
    if capture:
        return subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    subprocess.run(command, cwd=ROOT, check=True)


def rows_for(manifest: Path):
    with manifest.open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def needs_gemini(row: dict[str, str], refresh_confidence: bool) -> bool:
    return row["status"] == "unlabeled" or bool(
        refresh_confidence and row.get("gemini_model") and not row.get("gemini_confidence") and row["status"] != "blank"
    )


def reset_lite_proposals(manifest: Path):
    from src.scans.extract import write_manifest

    rows = rows_for(manifest)
    targets = [r for r in rows if r["gemini_model"] == "gemini-3.5-flash-lite" and
               r["status"] in ("proposed", "rejected") and r["label_source"] != "human_review"]
    if not targets:
        return 0
    backup = manifest.with_name(f"cells_before_flash_relabel_{datetime.now():%Y%m%d_%H%M%S}.csv")
    shutil.copy2(manifest, backup)
    for row in targets:
        row.update(status="unlabeled", label="", label_source="", gemini_label="", gemini_model="", gemini_confidence="")
    write_manifest(manifest, rows)
    print(f"Reset {len(targets)} Flash-Lite proposals for Flash; backup: {backup}", flush=True)
    return len(targets)


def paced_label(manifest: Path, model: str, min_interval: float, relabel_lite: bool, refresh_confidence: bool):
    if relabel_lite:
        reset_lite_proposals(manifest)
    if refresh_confidence:
        backup = manifest.with_name(f"cells_before_confidence_refresh_{datetime.now():%Y%m%d_%H%M%S}.csv")
        shutil.copy2(manifest, backup)
        print(f"Gemini confidence refresh backup: {backup}", flush=True)
    failures = 0
    while True:
        pending = sum(needs_gemini(r, refresh_confidence) for r in rows_for(manifest))
        if not pending:
            print("Gemini labeling complete", flush=True)
            return
        start = time.monotonic()
        options = [manifest, "--model", model, "--limit", "10"]
        if refresh_confidence:
            options.append("--refresh-confidence")
        result = run_script("label_scans_gemini.py", *options, capture=True)
        if result.stdout:
            print(result.stdout.rstrip(), flush=True)
        if result.returncode != 0:
            message = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "Unknown Gemini error"
            if "HTTP 429" in message and failures < 4:
                failures += 1
                delay = 60 * failures
                print(f"Gemini rate limit; waiting {delay}s before retry ({failures}/4)", flush=True)
                time.sleep(delay)
                continue
            raise RuntimeError(f"Labeling stopped: {message}")
        failures = 0
        remaining = sum(needs_gemini(r, refresh_confidence) for r in rows_for(manifest))
        if remaining >= pending:
            raise RuntimeError("Labeling did not advance; inspect the manifest")
        elapsed = time.monotonic() - start
        time.sleep(max(0, min_interval - elapsed))


def prepare(args):
    manifest = args.output / "cells.csv"
    if manifest.exists():
        print(f"Using existing extraction: {manifest}", flush=True)
        return manifest
    if not args.input:
        raise ValueError("--input is required when the output has no cells.csv")
    options = [args.input, "--output", args.output, "--cols", args.cols, "--rows", args.rows,
               "--writer-id", args.writer_id, "--size", args.size]
    if args.corners:
        options.extend(["--corners", *args.corners])
    run_script("prepare_scans.py", *options)
    return manifest


def run_ocr(manifest: Path, limit: int | None = None, overwrite: bool = False, include_blank: bool = False):
    options = [manifest]
    if limit is not None:
        options.extend(["--limit", limit])
    if overwrite:
        options.append("--overwrite")
    if include_blank:
        options.append("--include-blank")
    run_script("label_scans_ocr.py", *options)


def calibrate_and_prepare(args):
    manifest = args.output / "cells.csv"
    if manifest.exists():
        raise ValueError(f"{manifest} already exists; use a new output directory to preserve labels")
    command = [PYTHON, str(ROOT / "scripts" / "calibrate_scans_web.py"),
               "--input", str(args.input), "--output", str(args.output), "--port", str(args.port)]
    if args.cols is not None:
        command.extend(["--cols", str(args.cols)])
    if args.rows is not None:
        command.extend(["--rows", str(args.rows)])
    if args.no_browser:
        command.append("--no-browser")
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode == 2:
        print("Calibration cancelled; no cells were extracted", flush=True)
        return
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)
    saved = json.loads((args.output / "manual_calibration.json").read_text(encoding="utf-8"))
    corners = [coordinate for pair in saved["corners"] for coordinate in pair]
    run_script("prepare_scans.py", saved["source"], "--output", args.output,
               "--cols", saved["cols"], "--rows", saved["rows"],
               "--writer-id", args.writer_id, "--size", args.size, "--corners", *corners)


def review(manifest: Path, port: int, open_browser: bool):
    url = f"http://127.0.0.1:{port}/"
    print(f"Review in the browser: {url}; click '完成覆核' when finished", flush=True)
    command = [PYTHON, str(ROOT / "scripts" / "review_scans_web.py"), str(manifest), "--port", str(port)]
    process = subprocess.Popen(command, cwd=ROOT)
    try:
        time.sleep(0.5)
        if open_browser and process.poll() is None:
            webbrowser.open(url)
        process.wait()
    except KeyboardInterrupt:
        process.terminate()
        process.wait()
        print("Review server stopped", flush=True)


def export_package(manifest: Path, package: Path, include_proposed: bool):
    training = manifest.parent / ("train_manifest_proposals.csv" if include_proposed else "train_manifest.csv")
    options = [manifest, "--output", training]
    if include_proposed:
        options.append("--include-proposed")
    run_script("export_scanned_training.py", *options)
    run_script("package_scan_dataset.py", "pack", training, "--output", package)
    print(f"Git-trackable package ready: {package}, {package.with_suffix('.json')} and {package.with_name(package.stem + '_labels.jsonl')}", flush=True)


def train_package(args):
    cache = args.cache or (ROOT / "data" / "processed_scans" / "cloud_import" / args.package.stem)
    run_script("package_scan_dataset.py", "unpack", args.package, "--output", cache)
    command = [PYTHON, str(ROOT / "src" / "train.py"), "--dataset", "scanned",
               "--config", str(args.config), "--manifest", str(cache / "train_manifest.csv"),
               "--src_font", str(args.src_font), "--output_dir", str(args.output_dir),
               "--checkpoint_dir", str(args.checkpoint_dir)]
    if args.writer_id:
        command.extend(["--writer_id", args.writer_id])
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "prepare", "label", "ocr", "review", "report", "package"):
        command = sub.add_parser(name)
        command.add_argument("--output", type=Path, required=True, help="Working directory for extracted cells")
        if name in ("run", "prepare"):
            command.add_argument("--input", type=Path)
            command.add_argument("--cols", type=int, default=26)
            command.add_argument("--rows", type=int, default=22)
            command.add_argument("--size", type=int, default=128)
            command.add_argument("--writer-id", default="unknown")
            command.add_argument("--corners", type=float, nargs=8)
        if name in ("run", "label"):
            command.add_argument("--model", choices=("gemini-3.5-flash", "gemini-3.5-flash-lite"), default="gemini-3.5-flash")
            command.add_argument("--min-interval", type=float, default=4.0, help="Minimum seconds between request starts")
            command.add_argument("--relabel-lite", action="store_true", help="Back up and reprocess Flash-Lite proposals with Flash")
            command.add_argument("--refresh-confidence", action="store_true", help="Requery older Gemini cells without confidence; back up manifest")
        if name == "ocr":
            command.add_argument("--limit", type=int)
            command.add_argument("--overwrite", action="store_true")
            command.add_argument("--include-blank", action="store_true")
        if name in ("run", "review"):
            command.add_argument("--port", type=int, default=18765)
            command.add_argument("--no-browser", action="store_true")
        if name in ("run", "package"):
            command.add_argument("--package", type=Path)
            command.add_argument("--include-proposed", action="store_true")
        if name == "run":
            command.add_argument("--skip-review", action="store_true")
            command.add_argument("--skip-ocr", action="store_true")
    calibrate = sub.add_parser("calibrate", help="Choose grid corners and counts visually, then extract cells")
    calibrate.add_argument("--input", type=Path, required=True)
    calibrate.add_argument("--output", type=Path, required=True)
    calibrate.add_argument("--writer-id", default="unknown")
    calibrate.add_argument("--size", type=int, default=128)
    calibrate.add_argument("--cols", type=int, help="Initial column count; otherwise estimated from grid")
    calibrate.add_argument("--rows", type=int, help="Initial row count; otherwise estimated from grid")
    calibrate.add_argument("--port", type=int, default=18766)
    calibrate.add_argument("--no-browser", action="store_true")
    train = sub.add_parser("train")
    train.add_argument("--package", type=Path, required=True)
    train.add_argument("--src-font", type=Path, required=True)
    train.add_argument("--config", type=Path, default=Path("configs/scanned_unet_size128.yaml"))
    train.add_argument("--writer-id", default="")
    train.add_argument("--cache", type=Path)
    train.add_argument("--output-dir", type=Path, default=Path("runs/scanned"))
    train.add_argument("--checkpoint-dir", type=Path, default=Path("runs_checkpoints/scanned"))
    args = parser.parse_args()
    if args.command == "calibrate":
        return calibrate_and_prepare(args)
    if args.command == "train":
        return train_package(args)
    if args.command in ("run", "prepare"):
        manifest = prepare(args)
    else:
        manifest = args.output / "cells.csv"
        if not manifest.is_file():
            parser.error(f"Missing manifest: {manifest}")
    if args.command == "report":
        return run_script("report_scan_confidence.py", manifest)
    if args.command in ("run", "label"):
        if args.min_interval < 1:
            parser.error("--min-interval must be at least 1 second")
        if args.relabel_lite and args.model != "gemini-3.5-flash":
            parser.error("--relabel-lite requires --model gemini-3.5-flash")
        paced_label(manifest, args.model, args.min_interval, args.relabel_lite, args.refresh_confidence)
    if args.command == "ocr" or (args.command == "run" and not args.skip_ocr):
        run_ocr(manifest, getattr(args, "limit", None), getattr(args, "overwrite", False), getattr(args, "include_blank", False))
    if args.command in ("run", "review") and not (args.command == "run" and args.skip_review):
        review(manifest, args.port, not args.no_browser)
    if args.command in ("run", "package"):
        package = args.package or (ROOT / "data" / "scanned_packages" / f"{args.output.name}.npz")
        export_package(manifest, package, args.include_proposed)


if __name__ == "__main__":
    main()
