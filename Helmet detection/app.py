import streamlit as st
st.set_page_config(page_title="Helmet Detector", layout="centered")
import torch
from pathlib import Path
from PIL import Image
import os
import shutil
import tempfile
import cv2
import time

YOLOV5_DIR = Path("yolov5").resolve()
MODEL_PATH = YOLOV5_DIR / "runs/train/exp4/weights/best.pt"

# Load YOLOv5 model
@st.cache_resource
def load_model():
    return torch.hub.load(str(YOLOV5_DIR), 'custom', path=str(MODEL_PATH), source='local')

model = load_model()

st.title("🪖 Helmet Detection using YOLOv5")
st.markdown("Upload an image to detect if the person is wearing a **helmet** or **no helmet**.")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Save uploaded image temporarily
    temp_img_path = "temp_uploaded_img.jpg"
    with open(temp_img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display uploaded image
    st.image(Image.open(temp_img_path), caption="Uploaded Image", use_container_width=True)

    # Run detection
    st.write("🔍 Running detection...")
    results = model(temp_img_path)

    # Save results
    output_dir = Path("yolov5/runs/detect/streamlit")
    if output_dir.exists():
        shutil.rmtree(output_dir)  # clear previous
    results.save(save_dir=output_dir)

    # Show result
    result_img_path = list(output_dir.glob("*.jpg"))[0]
    st.image(Image.open(result_img_path), caption="Prediction Result", use_container_width=True)

    # Clean up temp file
    os.remove(temp_img_path)

st.markdown("---")
st.subheader("🎥 Video Detection")
video_file = st.file_uploader("Upload a video...", type=["mp4", "mov", "avi"], key="video")

if video_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    video_path = tfile.name

    # Load video
    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS))

    # Save output to temp file
    out_path = "output_video.mp4"
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    st.write("🔍 Processing video...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB and save as temp image
        temp_image_path = "frame.jpg"
        cv2.imwrite(temp_image_path, frame)

        # Run YOLOv5 on the frame
        results = model(temp_image_path)
        results.save(save_dir="yolov5/runs/detect/frame")

        # Load the saved prediction frame
        pred_frame_path = list(Path("yolov5/runs/detect/frame").glob("*.jpg"))[0]
        pred_frame = cv2.imread(str(pred_frame_path))

        # Write to video
        out.write(pred_frame)

        # Clean folder for next frame
        shutil.rmtree("yolov5/runs/detect/frame")

    cap.release()
    out.release()

    # Ensure file is flushed and readable
    time.sleep(2)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
        st.success("✅ Video processing complete!")

        # Convert for browser compatibility
        converted_path = out_path.replace(".mp4", "_converted.mp4")
        import subprocess

        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-i", out_path,
            "-vcodec", "libx264",
            "-crf", "23",
            converted_path
        ]

        result = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if result.returncode != 0:
            st.error("⚠️ FFmpeg conversion failed.")
            st.text(result.stderr.decode())
        elif os.path.exists(converted_path):
            st.success("🎉 Video converted successfully!")
            st.video(converted_path)
        else:
            st.warning("⚠️ Conversion completed, but output not found.")
   