"""
POST /detect  — image upload → detection + OCR response
POST /validate — cross-check detected prices against a reference price list
"""

from __future__ import annotations

import io
import json
import time
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from backend.app.schemas.detection import (
    BoundingBox, DetectionResponse, TagDetection,
    ValidationMismatch, ValidationResponse,
)
from backend.app.services.detector import get_detector
from backend.app.services.ocr_pipeline import read_price_from_crop

router = APIRouter()


def _load_image(upload: UploadFile) -> np.ndarray:
    """Read uploaded file into a BGR numpy array."""
    contents = upload.file.read()
    arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image. "
                            "Supported formats: JPEG, PNG, BMP, WebP.")
    return img


@router.post("/detect", response_model=DetectionResponse)
async def detect(
    file: UploadFile = File(..., description="Shelf image (JPEG/PNG)"),
    conf_threshold: float = Form(0.35, description="Detection confidence threshold"),
):
    """
    Detect price tags in the uploaded shelf image and extract prices via OCR.

    Returns bounding boxes, confidence scores, and extracted prices for each tag.
    Low-confidence OCR reads are flagged with `uncertain: true`.
    """
    t_start = time.perf_counter()

    img = _load_image(file)
    h_img, w_img = img.shape[:2]

    detector = get_detector()
    boxes = detector.detect(img, conf=conf_threshold)

    if not boxes:
        return DetectionResponse(
            image_width=w_img,
            image_height=h_img,
            tag_count=0,
            tags=[],
            processing_time_ms=round((time.perf_counter() - t_start) * 1000, 1),
            model_used=detector.model_name,
            ocr_engine="easyocr",
        )

    tags = []
    for idx, box in enumerate(boxes):
        crop = box.crop(img)

        if crop.size == 0:
            ocr_price, ocr_raw, ocr_conf, ocr_uncertain, ocr_prep = None, "", 0.0, True, "none"
        else:
            ocr = read_price_from_crop(crop)
            ocr_price = ocr.price
            ocr_raw = ocr.raw_text
            ocr_conf = ocr.confidence
            ocr_uncertain = ocr.uncertain
            ocr_prep = ocr.preprocessing

        tags.append(TagDetection(
            tag_id=idx,
            bounding_box=BoundingBox(
                x1=box.x1, y1=box.y1, x2=box.x2, y2=box.y2,
                cx=box.cx, cy=box.cy,
                width=box.width, height=box.height,
            ),
            detection_confidence=round(box.confidence, 4),
            price=ocr_price,
            raw_ocr_text=ocr_raw,
            ocr_confidence=round(ocr_conf, 4),
            uncertain=ocr_uncertain,
            ocr_preprocessing=ocr_prep,
        ))

    elapsed_ms = round((time.perf_counter() - t_start) * 1000, 1)

    return DetectionResponse(
        image_width=w_img,
        image_height=h_img,
        tag_count=len(tags),
        tags=tags,
        processing_time_ms=elapsed_ms,
        model_used=detector.model_name,
        ocr_engine="easyocr",
    )


@router.post("/validate", response_model=ValidationResponse)
async def validate(
    file: UploadFile = File(..., description="Shelf image"),
    price_list: UploadFile = File(..., description="JSON price list: [{sku, price}]"),
    conf_threshold: float = Form(0.35),
    tolerance: float = Form(0.01, description="Acceptable price difference (e.g. 0.01 = 1 cent)"),
):
    """
    Detect price tags, extract prices, then cross-check against a reference
    price list (JSON file). Flags any mismatches.

    Price list format (JSON array):
        [{"sku": "ITEM001", "price": 2.99}, ...]

    Since detected tags don't carry SKU info, validation compares the *set*
    of detected prices against the *set* of expected prices.
    """
    # Load reference prices
    try:
        ref_data = json.loads(await price_list.read())
        expected_prices = {float(item["price"]) for item in ref_data}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid price list JSON: {e}")

    # Rewind and detect
    await file.seek(0)
    detection_result = await detect(file=file, conf_threshold=conf_threshold)

    mismatches = []
    for tag in detection_result.tags:
        detected_val = float(tag.price) if tag.price else None
        flagged = False
        reason = "ok"

        if detected_val is None:
            flagged = True
            reason = "ocr_failed"
        elif tag.uncertain:
            flagged = True
            reason = "low_confidence"
        else:
            # Check if detected price matches any expected price within tolerance
            matched = any(abs(detected_val - exp) <= tolerance for exp in expected_prices)
            if not matched:
                flagged = True
                reason = "price_not_in_reference_list"

        # Find closest expected for reporting
        closest_expected = None
        diff = None
        if detected_val is not None and expected_prices:
            closest_expected = min(expected_prices, key=lambda p: abs(p - detected_val))
            diff = round(abs(detected_val - closest_expected), 4)

        mismatches.append(ValidationMismatch(
            tag_id=tag.tag_id,
            detected_price=tag.price,
            expected_price=str(closest_expected) if closest_expected else None,
            difference=diff,
            flagged=flagged,
            reason=reason,
        ))

    mismatch_count = sum(1 for m in mismatches if m.flagged)
    return ValidationResponse(
        tag_count=len(mismatches),
        mismatches=mismatches,
        mismatch_count=mismatch_count,
        all_ok=(mismatch_count == 0),
    )
