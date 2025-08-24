import streamlit as st
import requests
import tempfile
import cv2 # type: ignore
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean, median

# Hugging Face API settings
API_URL = "https://api-inference.huggingface.co/models/microsoft/VideoFakeDetection"
headers = {"Authorization": "Bearer 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"}  # 🔑 Replace with your key

st.set_page_config(page_title="Fake Video Detector", layout="wide")
st.title("🎥 Fake Video Detector")
st.write("Upload a video to check if it's **real or manipulated** using Hugging Face deepfake detection.")

# Upload video
video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

def query_api(video_bytes):
    response = requests.post(API_URL, headers=headers, data=video_bytes)
    return response.json()

if video_file:
    st.video(video_file)

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(video_file.read())
        video_path = tmp.name

    # Extract frames
    st.info("Extracting frames...")
    cap = cv2.VideoCapture(video_path)
    frames, frame_scores = [], []
    frame_count = 0

    progress = st.progress(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 15 == 0:  # sample every 15th frame
            _, buffer = cv2.imencode(".jpg", frame)
            response = query_api(buffer.tobytes())

            # Fake probability
            if isinstance(response, list) and "label" in response[0]:
                fake_prob = [r["score"] for r in response if r["label"] == "FAKE"]
                fake_prob = fake_prob[0] if fake_prob else 0
                frame_scores.append(fake_prob)
                frames.append(frame)

        progress.progress(min(frame_count / int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1.0))

    cap.release()

    if frame_scores:
        avg_score = mean(frame_scores)
        median_score = median(frame_scores)
        max_score = max(frame_scores)

        st.subheader("📊 Detection Results")
        st.write(f"**Average Fake Probability:** {avg_score:.2f}")
        st.write(f"**Median Fake Probability:** {median_score:.2f}")
        st.write(f"**Max Fake Probability:** {max_score:.2f}")

        verdict = "🚨 FAKE" if avg_score > 0.5 else "✅ REAL"
        st.markdown(f"### Final Verdict: **{verdict}**")

        # Histogram
        fig, ax = plt.subplots()
        ax.hist(frame_scores, bins=10, color="red", alpha=0.7, edgecolor="black")
        ax.set_title("Distribution of Fake Probabilities")
        ax.set_xlabel("Fake Probability")
        ax.set_ylabel("Frame Count")
        st.pyplot(fig)

        # Suspicious frames
        st.subheader("🔍 Most Suspicious Frames")
        top_indices = np.argsort(frame_scores)[-3:]
        cols = st.columns(3)
        for i, idx in enumerate(top_indices):
            with cols[i]:
                st.image(frames[idx], caption=f"Frame {idx} | Score: {frame_scores[idx]:.2f}", use_container_width=True)

    else:
        st.error("No fake probabilities detected. Try another video.")

