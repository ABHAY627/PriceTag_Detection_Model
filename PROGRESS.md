# PROGRESS — Retail Price Tag Detection & OCR

## Status Overview

| Checkpoint | Title                  | Status      |
|------------|------------------------|-------------|
| 1          | Data Preparation       | ✅ Done      |
| 2          | Detection Model        | ✅ Done      |
| 3          | OCR & Validation       | ✅ Done      |
| 4          | Integration Backend    | ✅ Done      |
| 5          | Frontend & Reporting   | ✅ Done      |

---

## Checkpoint 1 — Data Preparation ✅
- Repo structure scaffolded
- 20 unannotated shelf images found in `data/raw/Sample Dataset/`
- `scripts/prepare_dataset.py` — ingest, validate, split (70/15/15)
- `scripts/augment_dataset.py` — brightness/contrast, flip, rotate, blur, occlusion, tag shift
- `scripts/generate_labels_cv.py` — classical CV bootstrap annotation (no ML needed)
- `scripts/auto_annotate.py` — YOLOv8-COCO pretrained bootstrap (requires ultralytics)
- `data/dataset.yaml` — YOLO config, 1 class: price_tag

**Dataset note:** Images are unannotated. Run `generate_labels_cv.py --preview` to
generate bootstrap labels, then review/correct in LabelImg before training.

---

## Checkpoint 2 — Detection Model ✅
- `scripts/train_detector.py` — full YOLOv8 fine-tune script
- Supports yolov8n / yolov8s / yolov8m
- Early stopping (patience=20), saves best.pt + last.pt to `models/checkpoints/`
- Writes `DETECTION_REPORT.md` with mAP@0.5 and mAP@0.5:0.95 metrics

---

## Checkpoint 3 — OCR & Validation ✅
- `backend/app/services/ocr_pipeline.py`
  - EasyOCR adapter (default, no API key)
  - Google Vision adapter stub (activates via GOOGLE_VISION_KEY env var)
  - Two-pass processing: standard CLAHE → aggressive Otsu binarisation on retry
  - Regex price extraction with OCR confusion fixes (O/0, S/5, B/8, l/1)
  - Confidence threshold (0.55 default), uncertain flag for low-conf reads

---

## Checkpoint 4 — Integration Backend ✅
- `backend/app/main.py` — FastAPI app with CORS
- `POST /detect` — image → bounding boxes + prices + processing time
- `POST /validate` — image + price list JSON → mismatch flagging
- `GET /health` — health check
- MockDetector fallback when no trained weights exist (demo mode)
- `backend/sample_response.json` — example API response
- Auto-docs at http://localhost:8000/docs

---

## Checkpoint 5 — Frontend ✅
- Next.js 15 + TypeScript + Tailwind
- Retail/industrial aesthetic: near-black bg, shelf-label yellow accent, monospace prices
- Drag-drop image upload
- Canvas bounding-box overlay with scanner viewfinder corner brackets
- Right panel: per-tag price, confidence bars, uncertain blink indicator
- Collapsible full data table
- Export JSON button
- Confidence threshold slider in header
- `frontend/.env.local` → NEXT_PUBLIC_API_URL=http://localhost:8000

---

## Deployment ✅
- `Dockerfile` — Python 3.11 slim, downloads weights from HuggingFace Hub at build time
- `render.yaml` — Render Web Service config (Docker runtime, free tier)
- `frontend/vercel.json` — Vercel Next.js config
- `backend/app/services/model_loader.py` — auto-downloads best.pt from HF Hub on startup
- `DEPLOYMENT.md` — full step-by-step hosting guide

**Hosting plan:** Vercel (frontend) + Render (backend) + HuggingFace Hub (model weights)

---

## Open Questions for You
1. **Annotations** — Run `python scripts/generate_labels_cv.py --preview` to bootstrap labels.
   Review previews in `data/annotation_previews/`, correct in LabelImg, then run `prepare_dataset.py`.
2. **Compute** — For training: local GPU? Colab? Determines epochs/batch config.
3. **Deployment** — Vercel (frontend) + Render (backend)? Confirm before any deploy step.
4. **Google Vision OCR** — Not enabled. Set `GOOGLE_VISION_KEY` env var if/when you want it.
5. **Price list** for `/validate` endpoint — do you have a reference CSV/JSON to test with?
