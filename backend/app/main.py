"""
Retail Price Tag Detection & OCR — FastAPI Backend
===================================================
Run:
    uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000

API docs (auto-generated):
    http://localhost:8000/docs
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.routers.detect import router as detect_router
from backend.app.services.model_loader import ensure_weights


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Download model weights on startup if needed
    ensure_weights()
    yield


app = FastAPI(
    lifespan=lifespan,
    title="Retail Price Tag Detection & OCR",
    description=(
        "Upload a shelf image to detect price tags and extract numeric prices. "
        "Powered by YOLOv8 + EasyOCR."
    ),
    version="1.0.0",
)

# Allow Next.js frontend (localhost:3000) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(detect_router, tags=["Detection"])


@app.get("/health")
def health():
    return {"status": "ok", "service": "retail-price-tag-ocr"}
