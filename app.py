import streamlit as st
import cv2
import tempfile
import time
import numpy as np
from utils import process_frame

st.set_page_config(page_title="Lane Detection + Offset System", layout="wide")

st.title("🚗 Advanced Lane Detection with Vehicle Offset Monitoring")
st.markdown("Real-time lane detection with deviation analysis and drift warning.")

uploaded_file = st.file_uploader("📂 Upload MP4 Video", type=["mp4"])

# Session state for stopping safely
if "stop_processing" not in st.session_state:
    st.session_state.stop_processing = False

if uploaded_file is not None:

    col1, col2 = st.columns(2)

    with col1:
        start_button = st.button("🚀 Start Processing")

    with col2:
        stop_button = st.button("🛑 Stop Processing")

    if start_button:
        st.session_state.stop_processing = False

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
            tfile.write(uploaded_file.read())
            input_path = tfile.name

        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            st.error("❌ Could not open video.")
        else:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            current_frame = 0

            progress = st.progress(0)
            frame_placeholder = st.empty()
            offset_display = st.empty()
            alert_display = st.empty()

            st.info("🎥 Processing Video...")

            while True:

                # Stop logic
                if st.session_state.stop_processing:
                    alert_display.warning("⛔ Processing Stopped by User")
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                start_time = time.time()

                result, lane_positions = process_frame(frame)

                if result is None:
                    result = frame

                height, width, _ = frame.shape
                car_center = width // 2

                offset_text = "Lane Not Detected"
                alert_message = ""

                if lane_positions is not None:

                    left_x, right_x = lane_positions
                    lane_center = (left_x + right_x) // 2

                    offset_pixels = car_center - lane_center

                    # Pixel to meter approximation
                    xm_per_pix = 3.7 / 700
                    offset_meters = offset_pixels * xm_per_pix

                    direction = "Right" if offset_meters > 0 else "Left"

                    offset_text = f"{abs(offset_meters):.2f} m {direction}"

                    # Drift alert threshold
                    if abs(offset_meters) > 0.5:
                        alert_message = "⚠ Lane Departure Warning!"
                        cv2.putText(result, alert_message,
                                    (50, 120),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    (0, 0, 255),
                                    3)

                    # Draw car center (Blue)
                    cv2.line(result, (car_center, height),
                             (car_center, height-150),
                             (255, 0, 0), 3)

                    # Draw lane center (Green)
                    cv2.line(result, (lane_center, height),
                             (lane_center, height-150),
                             (0, 255, 0), 3)

                # FPS calculation
                fps = 1 / (time.time() - start_time + 0.00001)

                cv2.putText(result, f"FPS: {int(fps)}",
                            (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2)

                # Combine Original + Processed
                combined = cv2.hconcat([frame, result])

                frame_placeholder.image(combined, channels="BGR")

                offset_display.metric("Vehicle Offset From Center", offset_text)

                if alert_message:
                    alert_display.error(alert_message)
                else:
                    alert_display.success("✅ Vehicle Within Lane")

                current_frame += 1
                progress.progress(min(current_frame / total_frames, 1.0))

            cap.release()
            st.success("✅ Processing Complete!")

    if stop_button:
        st.session_state.stop_processing = True