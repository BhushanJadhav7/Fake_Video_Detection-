# Video Manipulation Detector (Hugging Face + Streamlit)

**Status:** Step 1 — Project setup + basic Streamlit app + Hugging Face Inference API test on frames.

This project detects manipulated videos (deepfakes/shallow fakes) by extracting frames and calling a hosted model on the Hugging Face Inference API.

We will iterate step-by-step. This README will be expanded in later steps with architecture diagrams, evaluation methodology, and deployment instructions.

---

## Quickstart

1) **Clone or unzip** this project.
2) **Create a virtual environment** (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```
3) **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4) **Set your Hugging Face API key**:
   - Copy `.env.example` to `.env` and set `HF_API_KEY` (and optionally `HF_MODEL_ID`).

5) **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```
   Upload a short video (mp4/avi/mov/mkv), extract a few frames, and test the API on a frame.

---

## Env Vars

- `HF_API_KEY`: Your Hugging Face token (Settings → Access Tokens).
- `HF_MODEL_ID` (optional): Model repo id. Default: `selimsef/xception`.

> Note: Many HF deepfake models are frame-based and may have different output formats. We include a parser that handles common patterns.

---

## Next Steps (Planned)

- Step 2: Full-frame batch analysis and aggregation per video.
- Step 3: Add face detection/cropping option for better signal.
- Step 4: Caching, progress bars, and better per-frame visualization.
- Step 5: Packaging, Dockerfile, and deployment notes.

