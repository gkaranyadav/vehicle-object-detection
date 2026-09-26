import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from PIL import Image
import io
import time
from ultralytics import YOLO

st.set_page_config(page_title="Vehicle & Object Analytics", page_icon="🚗", layout="wide")

# classes YOLO's coco weights already know about, just splitting them into
# two buckets so the dashboard/csv export makes more sense
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle", "train"}

MODEL_PATH = "yolov8n.pt"  # nano model, good enough and doesn't need a gpu


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


def run_detection(model, frame, conf_threshold=0.4):
    results = model(frame, conf=conf_threshold, verbose=False)[0]
    vehicles, other = [], []
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        conf = float(box.conf[0])
        # print(label, conf)  # was debugging class mapping earlier, leaving this here for now
        row = {"timestamp": now, "label": label, "confidence": round(conf, 3)}
        if label in VEHICLE_CLASSES:
            vehicles.append(row)
        else:
            other.append(row)

    annotated = results.plot()  # ultralytics draws the boxes for us, easier than doing it by hand
    return annotated, vehicles, other


def get_frame_source(source_type, rtsp_url, cam_index):
    # TODO: retry logic if the RTSP connection drops mid-stream, works fine for now though
    if source_type == "Webcam":
        return cv2.VideoCapture(cam_index)
    elif source_type == "RTSP / IP camera":
        return cv2.VideoCapture(rtsp_url)
    return None


def main():
    st.title("🚗 Vehicle & Object Detection")
    st.caption("Runs YOLOv8 locally — no cloud backend, works from any camera or uploaded image")

    if 'vehicles_data' not in st.session_state:
        st.session_state.vehicles_data = []
        st.session_state.other_data = []
        st.session_state.running = False
        # not persisting this anywhere yet, resets on refresh — fine for a demo

    model = load_model()

    with st.sidebar:
        st.header("Source")
        source_type = st.radio("Input", ["Webcam", "RTSP / IP camera", "Upload image"])

        rtsp_url = ""
        cam_index = 0
        if source_type == "RTSP / IP camera":
            rtsp_url = st.text_input("RTSP URL", placeholder="rtsp://username:password@ip:port/stream")
        elif source_type == "Webcam":
            cam_index = st.number_input("Camera index", min_value=0, max_value=5, value=0)

        conf_threshold = st.slider("Confidence threshold", 0.1, 0.9, 0.4, 0.05)

        st.divider()
        st.metric("Vehicles logged", len(st.session_state.vehicles_data))
        st.metric("Other objects logged", len(st.session_state.other_data))

    tab1, tab2 = st.tabs(["Live Detection", "Analytics & Export"])

    with tab1:
        if source_type == "Upload image":
            uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
            if uploaded:
                img = Image.open(uploaded).convert("RGB")
                frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                annotated, vehicles, other = run_detection(model, frame, conf_threshold)
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_column_width=True)
                st.session_state.vehicles_data.extend(vehicles)
                st.session_state.other_data.extend(other)
                if vehicles or other:
                    st.success(f"found {len(vehicles)} vehicles, {len(other)} other objects")
                else:
                    st.info("nothing detected above the confidence threshold")

        else:
            col1, col2 = st.columns(2)
            start = col1.button("Start", type="primary")
            stop = col2.button("Stop")

            if start:
                st.session_state.running = True
            if stop:
                st.session_state.running = False

            frame_slot = st.empty()
            info_slot = st.empty()

            if st.session_state.running:
                cap = get_frame_source(source_type, rtsp_url, cam_index)
                if not cap or not cap.isOpened():
                    st.error("couldn't open that video source — check the URL/index and try again")
                    st.session_state.running = False
                else:
                    last_log = 0
                    while st.session_state.running:
                        ret, frame = cap.read()
                        if not ret:
                            st.warning("lost the video feed")
                            break

                        annotated, vehicles, other = run_detection(model, frame, conf_threshold)
                        frame_slot.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_column_width=True)

                        # only log every couple seconds so the csv doesn't fill up with duplicates
                        now = time.time()
                        if now - last_log > 2 and (vehicles or other):
                            last_log = now
                            st.session_state.vehicles_data.extend(vehicles)
                            st.session_state.other_data.extend(other)
                            info_slot.info(f"logged {len(vehicles)} vehicles, {len(other)} objects this frame")

                        time.sleep(0.03)
                    cap.release()
            else:
                st.info("hit start to begin detection")

    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Vehicles")
            if st.session_state.vehicles_data:
                vdf = pd.DataFrame(st.session_state.vehicles_data)
                st.dataframe(vdf.tail(15), use_container_width=True)
                st.bar_chart(vdf["label"].value_counts())
            else:
                st.info("no vehicle detections yet")

        with col2:
            st.subheader("Other objects")
            if st.session_state.other_data:
                odf = pd.DataFrame(st.session_state.other_data)
                st.dataframe(odf.tail(15), use_container_width=True)
                st.bar_chart(odf["label"].value_counts())
            else:
                st.info("no other-object detections yet")

        st.divider()
        # yeah these column names are inconsistent with the ones up top, didn't bother renaming
        c1, c2, c3 = st.columns(3)
        if st.session_state.vehicles_data:
            c1.download_button(
                "Download vehicles CSV",
                pd.DataFrame(st.session_state.vehicles_data).to_csv(index=False),
                f"vehicles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
            )
        if st.session_state.other_data:
            c2.download_button(
                "Download objects CSV",
                pd.DataFrame(st.session_state.other_data).to_csv(index=False),
                f"objects_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
            )
        if c3.button("Clear log"):
            st.session_state.vehicles_data = []
            st.session_state.other_data = []
            st.rerun()


if __name__ == "__main__":
    main()
