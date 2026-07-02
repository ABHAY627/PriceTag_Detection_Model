"""
Pydantic schemas for the detection API.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    cx: float
    cy: float
    width: float
    height: float


class TagDetection(BaseModel):
    tag_id: int
    bounding_box: BoundingBox
    detection_confidence: float
    price: Optional[str]          # e.g. "2.99" — None if OCR failed
    raw_ocr_text: str
    ocr_confidence: float
    uncertain: bool               # True = low confidence, needs human review
    ocr_preprocessing: str        # "standard" | "aggressive" | "none"


class DetectionResponse(BaseModel):
    image_width: int
    image_height: int
    tag_count: int
    tags: list[TagDetection]
    processing_time_ms: float
    model_used: str
    ocr_engine: str


class ValidationMismatch(BaseModel):
    tag_id: int
    detected_price: Optional[str]
    expected_price: Optional[str]
    difference: Optional[float]
    flagged: bool
    reason: str


class ValidationResponse(BaseModel):
    tag_count: int
    mismatches: list[ValidationMismatch]
    mismatch_count: int
    all_ok: bool
