"""
Checkpoint 3 — OCR & Price Validation Pipeline
================================================
Clean adapter interface: cropped tag image in → validated price string out.

Default engine: EasyOCR (free, local, no API key).
Google Vision adapter is stubbed and activates only when GOOGLE_VISION_KEY
is set in environment.

Usage (standalone test):
    python -m backend.app.services.ocr_pipeline --image path/to/crop.jpg
"""

from __future__ import annotations

import os
import re
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

# ── OCR Engine Adapter ────────────────────────────────────────────────────────

class OCRResult:
    def __init__(self, raw_text: str, confidence: float):
        self.raw_text = raw_text
        self.confidence = confidence  # 0.0 – 1.0


class EasyOCRAdapter:
    """Default adapter — no API key required."""

    def __init__(self):
        try:
            import easyocr
        except ImportError:
            raise ImportError("Run: pip install easyocr")
        # Initialise reader once (slow first time — downloads model weights)
        self._reader = easyocr.Reader(["en"], gpu=False, verbose=False)

    def read(self, img: np.ndarray) -> list[OCRResult]:
        results = self._reader.readtext(img, detail=1)
        return [
            OCRResult(raw_text=text, confidence=float(conf))
            for (_, text, conf) in results
        ]


class GoogleVisionAdapter:
    """
    Stub adapter for Google Cloud Vision OCR.
    Only activated when GOOGLE_VISION_KEY env var is set.
    # TODO: needs GOOGLE_VISION_KEY — ask before enabling
    """

    def __init__(self):
        key = os.environ.get("GOOGLE_VISION_KEY")
        if not key:
            raise EnvironmentError(
                "GOOGLE_VISION_KEY not set. "
                "Set the env var to use Google Vision OCR."
            )
        # TODO: initialise google.cloud.vision client here
        raise NotImplementedError(
            "Google Vision adapter not yet wired. "
            "Provide GOOGLE_VISION_KEY to enable."
        )

    def read(self, img: np.ndarray) -> list[OCRResult]:
        raise NotImplementedError


def get_ocr_adapter():
    """Return the appropriate adapter based on environment."""
    if os.environ.get("GOOGLE_VISION_KEY"):
        try:
            return GoogleVisionAdapter()
        except NotImplementedError:
            pass
    return EasyOCRAdapter()


# ── Image Preprocessing ───────────────────────────────────────────────────────

def preprocess_for_ocr(img: np.ndarray, aggressive: bool = False) -> np.ndarray:
    """
    Enhance crop for OCR readability.
    aggressive=True: used on retry when first pass has low confidence.
    """
    # Upscale if too small
    h, w = img.shape[:2]
    if w < 200:
        scale = 200 / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    if not aggressive:
        # Standard: CLAHE for contrast + mild denoise
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        gray = cv2.fastNlMeansDenoising(gray, h=10)
    else:
        # Aggressive: binarise with Otsu + morphological close
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((2, 2), np.uint8)
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# ── Price Text Extraction ─────────────────────────────────────────────────────

# Handles: $1.99  £2.50  €0.99  1,299  5.00  1.5  etc.
# Common OCR confusions fixed before regex: O→0, S→5, B→8, l→1, I→1
_OCR_FIXES = str.maketrans({
    "O": "0", "o": "0",
    "S": "5", "s": "5",   # only in numeric context — applied after isolation
    "B": "8",
    "l": "1", "I": "1",
    ",": ".",              # comma-decimal normalisation
})

_PRICE_PATTERN = re.compile(
    r"""
    (?:[$£€¥₹])?           # optional currency symbol
    \s*
    (\d{1,5}               # integer part (up to 99999)
    (?:[.,]\d{1,2})?)      # optional decimal (1-2 digits)
    \s*
    (?:[$£€¥₹])?           # trailing currency (some tags)
    """,
    re.VERBOSE,
)


def extract_price(raw_text: str) -> Optional[str]:
    """
    Clean OCR text and extract the first plausible price string.
    Returns normalised price like '2.99' or None if nothing found.
    """
    # Fix common OCR character confusions
    cleaned = raw_text.strip().translate(_OCR_FIXES)
    # Remove noise characters except digits, dot, currency symbols
    cleaned = re.sub(r"[^0-9.$£€¥₹.,]", " ", cleaned)

    match = _PRICE_PATTERN.search(cleaned)
    if not match:
        return None

    price_str = match.group(1).replace(",", ".")
    try:
        value = float(price_str)
    except ValueError:
        return None

    # Sanity check: retail prices typically between 0.01 and 9999.99
    if not (0.01 <= value <= 9999.99):
        return None

    return f"{value:.2f}"


# ── Main Pipeline Function ────────────────────────────────────────────────────

@dataclass
class PriceReadResult:
    price: Optional[str]        # e.g. "2.99" or None
    raw_text: str               # raw OCR output
    confidence: float           # highest OCR confidence seen (0.0-1.0)
    uncertain: bool             # True if confidence < threshold or no price found
    preprocessing: str          # "standard" or "aggressive" (which pass succeeded)


CONFIDENCE_THRESHOLD = 0.55
_adapter: object = None  # lazy singleton


def _get_adapter():
    global _adapter
    if _adapter is None:
        _adapter = get_ocr_adapter()
    return _adapter


def read_price_from_crop(
    crop: np.ndarray,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> PriceReadResult:
    """
    Main entry point.
    crop: BGR numpy array of the detected tag region.
    Returns PriceReadResult.
    """
    adapter = _get_adapter()

    for attempt, aggressive in enumerate([False, True]):
        processed = preprocess_for_ocr(crop.copy(), aggressive=aggressive)
        results = adapter.read(processed)

        if not results:
            continue

        # Take the highest-confidence result that yields a valid price
        results_sorted = sorted(results, key=lambda r: r.confidence, reverse=True)

        for ocr_result in results_sorted:
            price = extract_price(ocr_result.raw_text)
            if price and ocr_result.confidence >= confidence_threshold:
                return PriceReadResult(
                    price=price,
                    raw_text=ocr_result.raw_text,
                    confidence=ocr_result.confidence,
                    uncertain=False,
                    preprocessing="aggressive" if aggressive else "standard",
                )

        # Didn't meet threshold — try aggressive preprocessing on retry
        if attempt == 0:
            continue

        # Both passes done — return best guess flagged as uncertain
        best = results_sorted[0]
        price = extract_price(best.raw_text)
        return PriceReadResult(
            price=price,
            raw_text=best.raw_text,
            confidence=best.confidence,
            uncertain=True,
            preprocessing="aggressive",
        )

    return PriceReadResult(
        price=None,
        raw_text="",
        confidence=0.0,
        uncertain=True,
        preprocessing="none",
    )


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test OCR pipeline on a single image")
    parser.add_argument("--image", required=True, help="Path to cropped tag image")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print(f"❌  Cannot read {args.image}")
        exit(1)

    result = read_price_from_crop(img)
    print(f"\nRaw text   : {result.raw_text!r}")
    print(f"Price      : {result.price}")
    print(f"Confidence : {result.confidence:.3f}")
    print(f"Uncertain  : {result.uncertain}")
    print(f"Preprocess : {result.preprocessing}")
