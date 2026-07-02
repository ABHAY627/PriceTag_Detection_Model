"""
Checkpoint 1 — Data Augmentation
==================================
Expands the training split with augmented copies.
Operations applied per image (configurable via CLI):
  - Random rotation  (±15° default, bounding boxes rotated too)
  - Brightness / contrast jitter
  - Synthetic tag duplication + shifting  (copies existing tags to new positions)
  - Occlusion simulation  (random black rectangles over tag regions)
  - Horizontal flip
  - Gaussian blur

Output goes to data/augmented/ and is *merged* into data/splits/train/
so the model sees both originals and augmented copies.

Usage:
    python scripts/augment_dataset.py [--copies 3] [--seed 42]
"""

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TRAIN_IMG_DIR = ROOT / "data" / "splits" / "train" / "images"
TRAIN_LBL_DIR = ROOT / "data" / "splits" / "train" / "labels"
AUG_DIR = ROOT / "data" / "augmented"


# ── YOLO label helpers ────────────────────────────────────────────────────────

def read_labels(path: Path) -> list[list[float]]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            parts = line.split()
            rows.append([int(parts[0])] + [float(v) for v in parts[1:]])
    return rows


def write_labels(path: Path, labels: list[list[float]]) -> None:
    lines = [
        f"{int(r[0])} {r[1]:.6f} {r[2]:.6f} {r[3]:.6f} {r[4]:.6f}"
        for r in labels
    ]
    path.write_text("\n".join(lines))


def yolo_to_xyxy(cx, cy, w, h, img_w, img_h):
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return x1, y1, x2, y2


def xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h):
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return cx, cy, w, h


def clip_yolo(cx, cy, w, h):
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w = max(0.01, min(1.0, w))
    h = max(0.01, min(1.0, h))
    return cx, cy, w, h


# ── Augmentation functions ────────────────────────────────────────────────────

def aug_brightness_contrast(img: np.ndarray, rng: random.Random) -> np.ndarray:
    alpha = rng.uniform(0.6, 1.4)   # contrast
    beta = rng.randint(-40, 40)     # brightness
    out = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    return out


def aug_hflip(img: np.ndarray, labels: list) -> tuple[np.ndarray, list]:
    flipped = cv2.flip(img, 1)
    new_labels = []
    for r in labels:
        cid, cx, cy, w, h = r
        new_labels.append([cid, 1.0 - cx, cy, w, h])
    return flipped, new_labels


def aug_rotate(img: np.ndarray, labels: list, rng: random.Random) -> tuple[np.ndarray, list]:
    angle = rng.uniform(-15, 15)
    h_img, w_img = img.shape[:2]
    cx_img, cy_img = w_img / 2, h_img / 2
    M = cv2.getRotationMatrix2D((cx_img, cy_img), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w_img, h_img), borderMode=cv2.BORDER_REFLECT)
    new_labels = []
    for r in labels:
        cid, cx, cy, bw, bh = r
        x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, bw, bh, w_img, h_img)
        corners = np.array([[x1, y1, 1], [x2, y1, 1], [x2, y2, 1], [x1, y2, 1]], dtype=np.float32)
        rot_corners = (M @ corners.T).T
        nx1, ny1 = rot_corners[:, 0].min(), rot_corners[:, 1].min()
        nx2, ny2 = rot_corners[:, 0].max(), rot_corners[:, 1].max()
        ncx, ncy, nbw, nbh = xyxy_to_yolo(nx1, ny1, nx2, ny2, w_img, h_img)
        new_labels.append([cid, *clip_yolo(ncx, ncy, nbw, nbh)])
    return rotated, new_labels


def aug_blur(img: np.ndarray, rng: random.Random) -> np.ndarray:
    ksize = rng.choice([3, 5])
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def aug_occlusion(img: np.ndarray, labels: list, rng: random.Random) -> np.ndarray:
    """Draw 1-3 random black rectangles to simulate partial occlusion."""
    out = img.copy()
    h_img, w_img = out.shape[:2]
    n_occ = rng.randint(1, 3)
    for _ in range(n_occ):
        rw = rng.randint(int(w_img * 0.05), int(w_img * 0.20))
        rh = rng.randint(int(h_img * 0.05), int(h_img * 0.20))
        rx = rng.randint(0, max(0, w_img - rw))
        ry = rng.randint(0, max(0, h_img - rh))
        cv2.rectangle(out, (rx, ry), (rx + rw, ry + rh), (0, 0, 0), -1)
    return out


def aug_synthetic_shift(img: np.ndarray, labels: list, rng: random.Random) -> tuple[np.ndarray, list]:
    """
    Duplicate a random existing tag patch and paste it at a new (non-overlapping) location.
    Adds the new box to labels.
    """
    if not labels:
        return img, labels
    out = img.copy()
    h_img, w_img = out.shape[:2]
    src = rng.choice(labels)
    cid, cx, cy, bw, bh = src
    x1, y1, x2, y2 = [int(v) for v in yolo_to_xyxy(cx, cy, bw, bh, w_img, h_img)]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)
    patch = out[y1:y2, x1:x2]
    if patch.size == 0:
        return img, labels
    pw, ph = x2 - x1, y2 - y1
    # Random new top-left
    max_nx = w_img - pw
    max_ny = h_img - ph
    if max_nx <= 0 or max_ny <= 0:
        return img, labels
    nx = rng.randint(0, max_nx)
    ny = rng.randint(0, max_ny)
    out[ny: ny + ph, nx: nx + pw] = patch
    ncx, ncy, nbw, nbh = xyxy_to_yolo(nx, ny, nx + pw, ny + ph, w_img, h_img)
    new_labels = list(labels) + [[cid, *clip_yolo(ncx, ncy, nbw, nbh)]]
    return out, new_labels


# ── Augmentation pipeline ─────────────────────────────────────────────────────

AUGMENTATIONS = [
    "brightness_contrast",
    "hflip",
    "rotate",
    "blur",
    "occlusion",
    "synthetic_shift",
]


def apply_random_augs(img: np.ndarray, labels: list, rng: random.Random) -> tuple[np.ndarray, list]:
    ops = rng.sample(AUGMENTATIONS, k=rng.randint(2, 4))
    for op in ops:
        if op == "brightness_contrast":
            img = aug_brightness_contrast(img, rng)
        elif op == "hflip":
            img, labels = aug_hflip(img, labels)
        elif op == "rotate":
            img, labels = aug_rotate(img, labels, rng)
        elif op == "blur":
            img = aug_blur(img, rng)
        elif op == "occlusion":
            img = aug_occlusion(img, labels, rng)
        elif op == "synthetic_shift":
            img, labels = aug_synthetic_shift(img, labels, rng)
    return img, labels


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Augment training data")
    parser.add_argument("--copies", type=int, default=3,
                        help="Number of augmented copies per training image")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not TRAIN_IMG_DIR.exists() or not any(TRAIN_IMG_DIR.iterdir()):
        print("❌  data/splits/train/images/ is empty.")
        print("    Run  python scripts/prepare_dataset.py  first.")
        return

    # Clear and recreate augmented dir
    if AUG_DIR.exists():
        shutil.rmtree(AUG_DIR)
    (AUG_DIR / "images").mkdir(parents=True)
    (AUG_DIR / "labels").mkdir(parents=True)

    imgs = sorted(TRAIN_IMG_DIR.glob("*"))
    imgs = [p for p in imgs if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]

    if not imgs:
        print("❌  No images found in train/images/.")
        return

    rng = random.Random(args.seed)
    total_aug = 0

    for img_path in imgs:
        lbl_path = TRAIN_LBL_DIR / img_path.with_suffix(".txt").name
        if not lbl_path.exists():
            print(f"  ⚠️  No label for {img_path.name}, skipping.")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  ⚠️  Could not read {img_path.name}, skipping.")
            continue
        labels = read_labels(lbl_path)

        for i in range(args.copies):
            aug_img, aug_labels = apply_random_augs(img.copy(), list(labels), rng)
            stem = f"{img_path.stem}_aug{i:03d}"
            out_img = AUG_DIR / "images" / f"{stem}.jpg"
            out_lbl = AUG_DIR / "labels" / f"{stem}.txt"
            cv2.imwrite(str(out_img), aug_img)
            write_labels(out_lbl, aug_labels)
            total_aug += 1

    print(f"\n✅  Generated {total_aug} augmented images → data/augmented/")

    # Merge into train split
    merged = 0
    for aug_img in (AUG_DIR / "images").glob("*.jpg"):
        aug_lbl = AUG_DIR / "labels" / aug_img.with_suffix(".txt").name
        shutil.copy2(aug_img, TRAIN_IMG_DIR / aug_img.name)
        if aug_lbl.exists():
            shutil.copy2(aug_lbl, TRAIN_LBL_DIR / aug_lbl.name)
        merged += 1

    print(f"📂  Merged {merged} augmented samples into data/splits/train/")
    print(f"    Total train images now: {len(list(TRAIN_IMG_DIR.glob('*')))}")
    print("\n✅  Augmentation complete. Proceed to Checkpoint 2.")


if __name__ == "__main__":
    main()
