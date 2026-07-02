# ── Retail Price Tag OCR — Backend Dockerfile ─────────────────────────────────
# Base: Python 3.11 slim (keeps image size down)
FROM python:3.11-slim

# System deps needed by OpenCV + EasyOCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source
COPY backend/ ./backend/
COPY models/ ./models/

# Download model weights from HuggingFace Hub at build time if HF_MODEL_REPO is set
# (set this as a build arg in Render dashboard)
ARG HF_MODEL_REPO=""
ARG HF_MODEL_FILE="best.pt"
RUN if [ -n "$HF_MODEL_REPO" ]; then \
      pip install --no-cache-dir huggingface_hub && \
      python -c "from huggingface_hub import hf_hub_download; \
                 hf_hub_download(repo_id='$HF_MODEL_REPO', \
                                 filename='$HF_MODEL_FILE', \
                                 local_dir='models/checkpoints')"; \
    fi

# Expose FastAPI port
EXPOSE 8000

# Start server
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
