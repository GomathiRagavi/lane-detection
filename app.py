import streamlit as st
import cv2
import tempfile
import numpy as np
import os
from utils import process_frame

st.set_page_config(page_title="Lane Detection + Offset System", layout="wide")

st.title("🚗 Advanced Lane Detection with Vehicle Offset Monitoring")
st.markdown("Upload video → Process → Compare Before & After")

uploaded_file = st.file_uploader("📂 Upload MP4 Video", type=["mp4"])

if "processed_video_path" not in st.session_state:
    st.session_state.processed_video_path = None

if uploaded_file is not None:

    col1, col2 = st.columns(2)

    with col1:
        start_button = st.button("🚀 Start Processing")

    with col2:
        show_button = st.button("🎥 Show Output Video")

    # -------------------------
    # PROCESS VIDEO FULLY
    # -------------------------

    if start_button:

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
            tfile.write(uploaded_file.read())
            input_path = tfile.name

        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            st.error("❌ Could not open video.")
        else:
            st.info("⏳ Processing Video... Please wait")

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # 🔥 Resize for cloud performance
            new_width = 640
            new_height = 360

            output_path = "processed_output.mp4"

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (new_width*2, new_height))

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            progress = st.progress(0)
            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Resize frame
                frame = cv2.resize(frame, (new_width, new_height))

                result, _ = process_frame(frame)

                if result is None:
                    result = frame

                # Combine original + processed
                combined = cv2.hconcat([frame, result])

                out.write(combined)

                frame_count += 1
                progress.progress(min(frame_count / total_frames, 1.0))

            cap.release()
            out.release()

            st.session_state.processed_video_path = output_path
            st.success("✅ Processing Complete!")

    # -------------------------
    # SHOW OUTPUT VIDEO
    # -------------------------

    if show_button:
        if st.session_state.processed_video_path and os.path.exists(st.session_state.processed_video_path):

            st.subheader("🎥 Before vs After Comparison")
            st.video(st.session_state.processed_video_path)

            with open(st.session_state.processed_video_path, "rb") as f:
                st.download_button(
                    "⬇ Download Processed Video",
                    f,
                    file_name="lane_detection_output.mp4"
                )
        else:
            st.warning("⚠ Please process the video first.")