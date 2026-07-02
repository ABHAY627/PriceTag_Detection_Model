# Deployment Guide — PriceTag OCR

## Architecture
```
Browser → Vercel (Next.js frontend) → Render (FastAPI backend + YOLOv8)
                                              ↑
                                    model weights from HuggingFace Hub
```

---

## Prerequisites
- GitHub account (free)
- Vercel account (free) — vercel.com
- Render account (free) — render.com
- HuggingFace account (free) — huggingface.co
- Git installed on your machine
- Trained model weights: `models/checkpoints/best.pt`

---

## PHASE 1 — Push code to GitHub

### Step 1: Create a GitHub repo
1. Go to https://github.com/new
2. Name it `pricetag-ocr` (or anything you like)
3. Set to **Public** or **Private** — either works
4. Do NOT initialise with README (you already have one)
5. Click **Create repository**

### Step 2: Push your local project
Open a terminal in `d:\MUSE_ASSIGNMENT` and run:

```bash
git init                          # already done if .git exists — skip if so
git add .
git commit -m "initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/pricetag-ocr.git
git push -u origin main
```

---

## PHASE 2 — Upload model weights to HuggingFace Hub

Your trained `best.pt` is too large for GitHub. Store it on HuggingFace instead.

### Step 1: Create a HuggingFace model repo
1. Go to https://huggingface.co/new-model
2. Model name: `pricetag-ocr-model`
3. Set to **Public** (free) or **Private** (needs HF token)
4. Click **Create model**

### Step 2: Upload best.pt
Option A — via the website:
1. Open your new HF model repo
2. Click **Files and versions** → **Add file** → **Upload files**
3. Drag `models/checkpoints/best.pt` into the upload area
4. Click **Commit changes**

Option B — via CLI (faster for large files):
```bash
pip install huggingface_hub
huggingface-cli login          # paste your HF token from hf.co/settings/tokens
huggingface-cli upload YOUR_HF_USERNAME/pricetag-ocr-model models/checkpoints/best.pt best.pt
```

### Note your repo ID
It will be: `YOUR_HF_USERNAME/pricetag-ocr-model`
You'll need this in the next phase.

---

## PHASE 3 — Deploy Backend to Render

### Step 1: Create a Render account
1. Go to https://render.com and sign up (free)
2. Connect your GitHub account when prompted

### Step 2: Create a new Web Service
1. Click **New** → **Web Service**
2. Connect your GitHub repo (`pricetag-ocr`)
3. Fill in these settings:

| Field | Value |
|-------|-------|
| Name | `pricetag-ocr-backend` |
| Region | Oregon (US West) |
| Branch | `main` |
| Runtime | **Docker** |
| Dockerfile Path | `./Dockerfile` |
| Plan | **Free** |

4. Click **Advanced** → **Add Environment Variables**:

| Key | Value |
|-----|-------|
| `HF_MODEL_REPO` | `YOUR_HF_USERNAME/pricetag-ocr-model` |
| `HF_MODEL_FILE` | `best.pt` |

5. Click **Create Web Service**

### Step 3: Wait for deploy
- First deploy takes 10-15 minutes (installs PyTorch, EasyOCR, downloads model)
- Watch the logs in the Render dashboard
- When you see `Application startup complete` → it's live

### Step 4: Copy your backend URL
It will look like: `https://pricetag-ocr-backend.onrender.com`
Save this — you need it for the frontend.

> ⚠️ Free tier note: Render free services spin down after 15 min of inactivity.
> First request after sleep takes ~30s to wake up. Upgrade to Starter ($7/mo) to avoid this.

---

## PHASE 4 — Deploy Frontend to Vercel

### Step 1: Create a Vercel account
1. Go to https://vercel.com and sign up with GitHub (free)

### Step 2: Import your project
1. Click **Add New** → **Project**
2. Find and select your `pricetag-ocr` GitHub repo
3. Vercel will auto-detect Next.js

### Step 3: Configure the project
| Field | Value |
|-------|-------|
| Framework Preset | Next.js (auto-detected) |
| Root Directory | `frontend` ← IMPORTANT: set this |
| Build Command | `npm run build` |
| Output Directory | `.next` |

### Step 4: Add environment variable
Click **Environment Variables** and add:

| Key | Value |
|-----|-------|
| `NEXT_PUBLIC_API_URL` | `https://pricetag-ocr-backend.onrender.com` |

(Use your actual Render URL from Phase 3 Step 4)

### Step 5: Deploy
1. Click **Deploy**
2. Takes ~2 minutes
3. You'll get a live URL like: `https://pricetag-ocr.vercel.app`

---

## PHASE 5 — Verify it works

1. Open your Vercel URL in a browser
2. The app should load with the dark scanner UI
3. Test the backend health check:
   ```
   https://pricetag-ocr-backend.onrender.com/health
   ```
   Should return: `{"status":"ok","service":"retail-price-tag-ocr"}`

4. Try uploading a shelf image in the UI
5. Check API docs at:
   ```
   https://pricetag-ocr-backend.onrender.com/docs
   ```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Frontend shows "SCAN ERROR" | Check NEXT_PUBLIC_API_URL is set correctly in Vercel |
| Backend won't start | Check Render logs — likely OOM on free tier (PyTorch is heavy) |
| Backend returns 500 | Check HF_MODEL_REPO env var is correct |
| Render deploy times out | Free tier has build time limits — retry or upgrade |
| CORS error in browser | Backend URL in env var must match exactly (no trailing slash) |

---

## Updating after changes

**Frontend update:**
```bash
git add frontend/
git commit -m "update frontend"
git push
```
Vercel auto-deploys on every push to main.

**Backend update:**
```bash
git add backend/ Dockerfile requirements.txt
git commit -m "update backend"
git push
```
Render auto-deploys on every push to main.

**New model weights:**
Re-upload `best.pt` to HuggingFace Hub, then trigger a manual redeploy on Render
(Dashboard → your service → **Manual Deploy** → **Deploy latest commit**).
