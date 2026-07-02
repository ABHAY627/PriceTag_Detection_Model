"""
Checkpoint 2 — YOLOv8 Price Tag Detector Training
===================================================
Fine-tunes YOLOv8 on the prepared dataset.

Usage:
    # CPU / no GPU (small dataset, demo mode):
    python scripts/train_detector.py --model yolov8n --epochs 50 --batch 8 --imgsz 640

    # With GPU:
    python scripts/train_detector.py --model yolov8s --epochs 100 --batch 16 --imgsz 640

    # Quick smoke-test (2 epochs):
    python scripts/train_detector.py --epochs 2 --batch 4

Outputs:
    models/checkpoints/best.pt       ← best weights (by val mAP)
    models/checkpoints/last.pt       ← final epoch weights
    DETECTION_REPORT.md              ← metrics summary
"""

import argparse
import json
import shutil
from pathlib import Path
from datetime import datetime

try:
    from ultralytics import YOLO
except ImportError:
    print("❌  Run: pip install ultralytics")
    exit(1)

ROOT = Path(__file__).resolve().parent.parent
DATASET_YAML = ROOT / "data" / "dataset.yaml"
CHECKPOINTS_DIR = ROOT / "models" / "checkpoints"
DETECTION_REPORT = ROOT / "DETECTION_REPORT.md"


def write_detection_report(results, model_name: str, epochs: int, imgsz: int) -> None:
    metrics = results.results_dict if hasattr(results, "results_dict") else {}

    map50 = metrics.get("metrics/mAP50(B)", "—")
    map5095 = metrics.get("metrics/mAP50-95(B)", "—")
    precision = metrics.get("metrics/precision(B)", "—")
    recall = metrics.get("metrics/recall(B)", "—")

    def fmt(v):
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    lines = [
        "# DETECTION_REPORT",
        "",
        f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}_",
        "",
        "## Training Configuration",
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| Base model | {model_name}.pt |",
        f"| Epochs | {epochs} |",
        f"| Image size | {imgsz}px |",
        f"| Dataset | data/dataset.yaml |",
        "",
        "## Validation Metrics",
        "| Metric | Value |",
        "|--------|-------|",
        f"| mAP@0.5 | {fmt(map50)} |",
        f"| mAP@0.5:0.95 | {fmt(map5095)} |",
        f"| Precision | {fmt(precision)} |",
        f"| Recall | {fmt(recall)} |",
        "",
        "## Model Outputs",
        "- `models/checkpoints/best.pt` — best checkpoint (use this for inference)",
        "- `models/checkpoints/last.pt` — last epoch",
        "",
        "## Notes",
        "- Training run artifacts (curves, confusion matrix) saved in `runs/detect/`.",
        "- For better accuracy: add more annotated data and re-run with `--epochs 100`.",
    ]
    DETECTION_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n📄  DETECTION_REPORT.md written → {DETECTION_REPORT}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8 price tag detector")
    parser.add_argument("--model", default="yolov8n",
                        choices=["yolov8n", "yolov8s", "yolov8m"],
                        help="Base YOLO model size")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Training epochs (50 for CPU, 100+ for GPU)")
    parser.add_argument("--batch", type=int, default=8,
                        help="Batch size (8 for CPU, 16 for GPU)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size in pixels")
    parser.add_argument("--device", default="",
                        help="Device: '' = auto, 'cpu', '0' = GPU 0")
    args = parser.parse_args()

    if not DATASET_YAML.exists():
        print(f"❌  {DATASET_YAML} not found. Run prepare_dataset.py first.")
        return

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"🚀  Loading {args.model}.pt …")
    model = YOLO(f"{args.model}.pt")

    print(f"🏋️  Training for {args.epochs} epochs, batch={args.batch}, imgsz={args.imgsz} …")
    train_kwargs = dict(
        data=str(DATASET_YAML),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=str(ROOT / "runs" / "detect"),
        name="price_tag_v1",
        exist_ok=True,
        patience=20,          # early stopping
        save=True,
        val=True,
        verbose=True,
    )
    if args.device:
        train_kwargs["device"] = args.device

    results = model.train(**train_kwargs)

    # Copy best/last weights to models/checkpoints/
    run_dir = ROOT / "runs" / "detect" / "price_tag_v1" / "weights"
    for w_name in ["best.pt", "last.pt"]:
        src = run_dir / w_name
        if src.exists():
            shutil.copy2(src, CHECKPOINTS_DIR / w_name)
            print(f"  ✓  Saved {w_name} → models/checkpoints/")

    # Evaluate on val set
    print("\n📊  Evaluating on validation set …")
    val_results = model.val(data=str(DATASET_YAML))

    write_detection_report(val_results, args.model, args.epochs, args.imgsz)

    print("\n✅  Checkpoint 2 — training complete.")
    print("    Best weights: models/checkpoints/best.pt")
    print("    Next: python scripts/run_ocr_pipeline.py")


if __name__ == "__main__":
    main()
