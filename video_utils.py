import cv2
import numpy as np
from PIL import Image
import io

def extract_frames(video_path, fps=1, max_frames=60, output_size=(256, 256)):
    """Extract frames from a video at approx `fps` (frames per second).
    Returns a list of RGB numpy arrays of size output_size.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if not orig_fps or orig_fps <= 0:
        orig_fps = 30.0
    # interval in frames between samples
    frame_interval = max(1, int(round(orig_fps / float(fps))))

    frames = []
    idx = 0
    saved = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if idx % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if output_size:
                frame_rgb = cv2.resize(frame_rgb, output_size, interpolation=cv2.INTER_AREA)
            frames.append(frame_rgb)
            saved += 1
            if saved >= max_frames:
                break
        idx += 1

    cap.release()
    return frames

def encode_frame_to_jpeg_bytes(frame_rgb, quality=95):
    """Encode an RGB numpy array frame to JPEG bytes."""
    img = Image.fromarray(frame_rgb)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()
