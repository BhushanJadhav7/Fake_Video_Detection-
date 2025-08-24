import streamlit as st
import cv2 # type: ignore
import os
import tempfile
import requests
import numpy as np

# Hugging Face API
API_URL = "https://api-inference.huggingface.co/models/HuggingFaceM4/roop-fake-detection"
headers = {"Authorization": f"Bearer YOUR_HUGGINGFACE_API_KEY"}

st.title("🎥 Fake Video Detector - Step 4: Aggregation & Verdict")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

def query_image(image):
    _, img_encoded = cv2.imencode('.jpg', image)
    response = requests.post(API_URL, headers=headers, data=img_encoded.tobytes())
    return response.json()

if uploaded_file:
    # Save video to temp file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    st.success("✅ Video uploaded successfully!")

    # Extract frames (1 frame per second)
    vidcap = cv2.VideoCapture(tfile.name)
    frame_rate = int(vidcap.get(cv2.CAP_PROP_FPS))
    success, image = vidcap.read()
    count = 0
    frames = []

    while success:
        if count % frame_rate == 0:
            frames.append(image)
        success, image = vidcap.read()
        count += 1

    st.write(f"Extracted {len(frames)} frames for analysis.")

    fake_probs = []
    st.subheader("🖼️ Frame Analysis")

    for i, frame in enumerate(frames[:10]):  # limit to 10 frames for speed
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"Frame {i+1}")
        result = query_image(frame)

        # Parse result (depends on Hugging Face model output format)
        try:
            probs = {item["label"]: item["score"] for item in result}
            fake_prob = probs.get("fake", 0)
            fake_probs.append(fake_prob)
            st.write(f"Prediction: {probs}")
        except:
            st.error(f"Error analyzing frame {i+1}: {result}")

    # Aggregate results
    if fake_probs:
        avg_fake = np.mean(fake_probs) * 100
        st.subheader("📊 Final Verdict")
        st.write(f"Average Fake Probability: **{avg_fake:.2f}%**")

        if avg_fake > 70:
            st.error("❌ This video is **Likely FAKE**")
        elif avg_fake > 40:
            st.warning("⚠️ This video looks **Suspicious**")
        else:
            st.success("✅ This video is **Likely REAL**")
