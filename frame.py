import streamlit as st
import cv2
import os
import tempfile

st.title("🎥 Fake Video Detector - Step 2: Frame Extraction")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file:
    # Save video to temp file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    st.success("✅ Video uploaded successfully!")

    # Extract frames
    vidcap = cv2.VideoCapture(tfile.name)
    frame_rate = int(vidcap.get(cv2.CAP_PROP_FPS))  # frames per second
    success, image = vidcap.read()
    count = 0
    frames = []

    while success:
        # extract 1 frame per second
        if count % frame_rate == 0:
            frames.append(image)
        success, image = vidcap.read()
        count += 1

    st.write(f"Extracted {len(frames)} frames from the video.")

    # Show first 5 frames as preview
    st.subheader("🖼️ Frame Preview")
    for i, frame in enumerate(frames[:5]):
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"Frame {i+1}")
