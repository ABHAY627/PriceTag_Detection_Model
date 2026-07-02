"""
Checkpoint 1 — Dataset Preparation
====================================
Reads raw images + YOLO-format annotations from data/raw/,
validates them, then builds a reproducible train/val/test split
in data/splits/.

Usage:
    python scripts/prepare_dataset.py [--split 0.70 0.15 0.15] [--seed 42]

Annotation format expected in data/raw/:
    Each image file (jpg/jpeg/png/bmp/webp) should have a matching .txt file
    in the same directory with YOLO-format labels:
        <class_id> <cx> <cy> <w> <h>   (all values 0-1, normalised)

    If NO .txt file exists for an image the script will flag it and ask
    whether to skip unannotated images or abort.
"""

import argparse
import hashlib
import random
import shutil
import sys
from pathlib import Path

# ── Repo root is one level above /scripts ────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
SPLITS_DIR = ROOT / "data" / "splits"
DATA_REPORT = ROOT / "DATA_REPORT.md"

SUPPORTED_IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def md5(path: Path) -> str:
    h = hashlib.md5()
    h.update(path.read_bytes())
    return h.hexdigest()


def collect_pairs(raw_dir: Path) -> tuple[list[tuple[Path, Path]], list[Path]]:
    """Return (annotated_pairs, unannotated_images).
    Searches recursively so subfolders like 'Sample Dataset/' are handled.
    """
    annotated, unannotated = [], []
    for img in sorted(raw_dir.rglob("*")):
        if img.suffix.lower() not in SUPPORTED_IMG:
            continue
        label = img.with_suffix(".txt")
        if label.exists():
            annotated.append((img, label))
        else:
            unannotated.append(img)
    return annotated, unannotated


def validate_label(label_path: Path, class_count: int) -> list[str]:
    """Return list of error strings (empty = OK)."""
    errors = []
    for i, line in enumerate(label_path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"  line {i}: expected 5 values, got {len(parts)}")
            continue
        try:
            cid, cx, cy, w, h = int(parts[0]), *map(float, parts[1:])
        except ValueError:
            errors.append(f"  line {i}: non-numeric values")
            continue
        if cid >= class_count:
            errors.append(f"  line {i}: class_id {cid} >= nc={class_count}")
        for val, name in zip([cx, cy, w, h], ["cx", "cy", "w", "h"]):
            if not (0.0 <= val <= 1.0):
                errors.append(f"  line {i}: {name}={val} out of [0,1]")
    return errors


def split_pairs(
    pairs: list, ratios: tuple[float, float, float], seed: int
) -> tuple[list, list, list]:
    random.seed(seed)
    shuffled = list(pairs)
    random.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    return shuffled[:n_train], shuffled[n_train : n_train + n_val], shuffled[n_train + n_val :]


def copy_split(pairs: list, split_name: str, splits_dir: Path) -> None:
    img_dir = splits_dir / split_name / "images"
    lbl_dir = splits_dir / split_name / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for img, lbl in pairs:
        shutil.copy2(img, img_dir / img.name)
        shutil.copy2(lbl, lbl_dir / lbl.name)


def write_data_report(
    total: int,
    unannotated: int,
    invalid: int,
    duplicates: int,
    class_counts: dict,
    split_counts: dict,
    report_path: Path,
) -> None:
    lines = [
        "# DATA_REPORT",
        "",
        "## Summary",
        f"- Total images found in `data/raw/`: **{total + unannotated}**",
        f"- Annotated images: **{total}**",
        f"- Unannotated (skipped): **{unannotated}**",
        f"- Invalid annotation files (skipped): **{invalid}**",
        f"- Duplicate images removed: **{duplicates}**",
        f"- Final usable samples: **{total - invalid - duplicates}**",
        "",
        "## Split Distribution",
        "| Split | Images |",
        "|-------|--------|",
    ]
    for split, count in split_counts.items():
        lines.append(f"| {split} | {count} |")
    lines += [
        "",
        "## Class Balance",
        "| Class ID | Class Name | Instances |",
        "|----------|------------|-----------|",
    ]
    for cid, (name, count) in class_counts.items():
        lines.append(f"| {cid} | {name} | {count} |")
    lines += [
        "",
        "## Issues Found",
        "_See console output from prepare_dataset.py for per-file details._",
        "",
        "## Next Steps",
        "1. If unannotated images exist, annotate them with [LabelImg](https://github.com/HumanSignal/labelImg) or [Roboflow](https://roboflow.com).",
        "2. Run `python scripts/augment_dataset.py` to expand the dataset.",
        "3. Verify `data/dataset.yaml` class names match your annotation IDs.",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n📄  DATA_REPORT.md written → {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare YOLO dataset from data/raw/")
    parser.add_argument("--split", nargs=3, type=float, default=[0.70, 0.15, 0.15],
                        metavar=("TRAIN", "VAL", "TEST"),
                        help="Train/val/test split ratios (must sum to 1.0)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nc", type=int, default=1,
                        help="Number of classes (for label validation)")
    parser.add_argument("--class-names", nargs="+", default=["price_tag"],
                        help="Class names in class-id order")
    args = parser.parse_args()

    ratios = tuple(args.split)
    if abs(sum(ratios) - 1.0) > 1e-6:
        sys.exit("❌  Split ratios must sum to 1.0")

    if not RAW_DIR.exists() or not any(RAW_DIR.iterdir()):
        sys.exit(
            "❌  data/raw/ is empty.\n"
            "    Drop your images (and .txt label files) into data/raw/ first.\n"
            "    See PROGRESS.md → Checkpoint 1 open questions."
        )

    print("🔍  Scanning data/raw/ …")
    annotated, unannotated = collect_pairs(RAW_DIR)

    if unannotated:
        print(f"\n⚠️   {len(unannotated)} image(s) have NO matching .txt annotation file:")
        for p in unannotated[:10]:
            print(f"     {p.name}")
        if len(unannotated) > 10:
            print(f"     … and {len(unannotated) - 10} more.")
        resp = input("\nSkip unannotated images and continue? [y/N] ").strip().lower()
        if resp != "y":
            sys.exit("Aborted. Annotate missing files then re-run.")

    print(f"\n✅  Found {len(annotated)} annotated image(s).")

    # Validate labels
    invalid_pairs = []
    for img, lbl in annotated:
        errors = validate_label(lbl, args.nc)
        if errors:
            print(f"  ⚠️  {lbl.name} has errors:")
            for e in errors:
                print(e)
            invalid_pairs.append((img, lbl))

    valid_pairs = [p for p in annotated if p not in invalid_pairs]
    print(f"  → {len(valid_pairs)} valid / {len(invalid_pairs)} invalid (skipped)")

    # Deduplicate by image hash
    seen_hashes: dict[str, Path] = {}
    deduped, dup_count = [], 0
    for img, lbl in valid_pairs:
        h = md5(img)
        if h in seen_hashes:
            print(f"  🗑️  Duplicate: {img.name} matches {seen_hashes[h].name}")
            dup_count += 1
        else:
            seen_hashes[h] = img
            deduped.append((img, lbl))

    print(f"  → {dup_count} duplicate(s) removed. {len(deduped)} remain.")

    if not deduped:
        sys.exit("❌  No valid, unique samples to split. Check your data/raw/ contents.")

    # Split
    train, val, test = split_pairs(deduped, ratios, args.seed)
    print(f"\n📂  Split → train:{len(train)}  val:{len(val)}  test:{len(test)}")

    # Clear and copy
    if SPLITS_DIR.exists():
        shutil.rmtree(SPLITS_DIR)
    for split_name, pairs in [("train", train), ("val", val), ("test", test)]:
        copy_split(pairs, split_name, SPLITS_DIR)
        print(f"  ✓  {split_name}/  ({len(pairs)} samples)")

    # Count class instances
    class_counts: dict[int, list] = {
        i: [name, 0] for i, name in enumerate(args.class_names)
    }
    for _, lbl in deduped:
        for line in lbl.read_text().splitlines():
            line = line.strip()
            if line:
                cid = int(line.split()[0])
                if cid in class_counts:
                    class_counts[cid][1] += 1

    write_data_report(
        total=len(annotated),
        unannotated=len(unannotated),
        invalid=len(invalid_pairs),
        duplicates=dup_count,
        class_counts=class_counts,
        split_counts={"train": len(train), "val": len(val), "test": len(test)},
        report_path=DATA_REPORT,
    )

    print("\n✅  Checkpoint 1 — dataset preparation complete.")
    print("    Next: run  python scripts/augment_dataset.py")


if __name__ == "__main__":
    main()
