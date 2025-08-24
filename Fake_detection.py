import streamlit as st
import cv2
import os
import tempfile
import requests
import base64

# Hugging Face API
API_URL = "https://api-inference.huggingface.co/models/HuggingFaceM4/roop-fake-detection"
headers = {"Authorization": f"Bearer YOUR_HUGGINGFACE_API_KEY"}

st.title("🎥 Fake Video Detector - Step 3: Fake Detection")

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

    # Extract frames
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

    st.write(f"Extracted {len(frames)} frames from the video.")

    # Show first 5 frames + run detection
    st.subheader("🖼️ Detection Results on Sample Frames")
    for i, frame in enumerate(frames[:5]):
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"Frame {i+1}")
        result = query_image(frame)
        st.write(f"Prediction for Frame {i+1}: ", result)
