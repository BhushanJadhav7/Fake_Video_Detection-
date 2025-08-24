# 🎥 Fake Video Detector

A Streamlit app that detects **fake/manipulated videos** using the Hugging Face model [microsoft/VideoFakeDetection](https://huggingface.co/microsoft/VideoFakeDetection).

## 🚀 Features
- Upload video files (`.mp4`, `.avi`, `.mov`, `.mkv`)
- Frame-by-frame analysis via Hugging Face API
- Calculates average, median, and max fake probabilities
- Histogram of frame-wise fake scores
- Highlights the 3 most suspicious frames
- Final **Real vs Fake verdict**

## 🛠️ Installation
1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/fake-video-detector.git
   cd fake-video-detector


- Step 2: Full-frame batch analysis and aggregation per video.
- Step 3: Add face detection/cropping option for better signal.
- Step 4: Caching, progress bars, and better per-frame visualization.
- Step 5: Packaging, Dockerfile, and deployment notes.

