import streamlit as st
try:
    from ultralytics import YOLO
    _ULTRALYTICS_OK = True
except Exception as e:
    _ULTRALYTICS_OK = False
    _ULTRALYTICS_ERR = e
import cv2
import numpy as np
from PIL import Image
import time
from threading import Thread
try:
    import winsound  # Windows built-in beep
    _BEEP = True
except Exception:
    _BEEP = False

# Alarm/label configuration
ALERT_CLASSES = {"cigarette", "vape"}  # adjust to your class names
ALARM_COOLDOWN = 1.5  # seconds
if 'last_alarm' not in st.session_state:
    st.session_state.last_alarm = 0.0


def _play_alarm():
    if _BEEP:
        try:
            # 880 Hz for 250 ms
            winsound.Beep(880, 250)
        except Exception:
            pass


def _maybe_alarm():
    now = time.time()
    if now - st.session_state.last_alarm >= ALARM_COOLDOWN:
        Thread(target=_play_alarm, daemon=True).start()
        st.session_state.last_alarm = now


def _extract_detected_names_v8(results):
    try:
        r = results[0]
        names_map = r.names  # id -> name
        clses = r.boxes.cls.cpu().numpy().astype(int) if hasattr(r, 'boxes') and r.boxes is not None else []
        names = [str(names_map.get(int(c), str(c))).lower() for c in clses]
        return names
    except Exception:
        return []


def annotate_and_alert_v8(results, frame_bgr):
    # Create annotated frame from YOLOv8
    annotated = results[0].plot()
    names = _extract_detected_names_v8(results)
    found = sorted({n for n in names if any(t in n for t in ALERT_CLASSES)})

    if found:
        label_text = f"ALERT: {', '.join([n.title() for n in found])}"
        # Draw filled rectangle banner top-left
        pad_w = 24
        text_scale = 0.8
        text_thickness = 2
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness)
        x1, y1 = 10, 10
        x2, y2 = x1 + tw + pad_w, y1 + th + 30
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 0), -1)
        cv2.putText(annotated, label_text, (x1 + 10, y1 + th + 10), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 0, 255), text_thickness, cv2.LINE_AA)
        _maybe_alarm()

    return annotated

# Load YOLO model
@st.cache_resource
def load_model():
    if not _ULTRALYTICS_OK:
        raise RuntimeError(f"Ultralytics import failed: {_ULTRALYTICS_ERR}. Please install in the same Python env running Streamlit.")
    return YOLO("best.pt")

if _ULTRALYTICS_OK:
    model = load_model()
else:
    st.error("Ultralytics is not installed in the Streamlit environment. See instructions below.")

st.title("Cigarette & Vape Detection")
if not _ULTRALYTICS_OK:
    st.info(
        "Ultralytics isn't available in the Python environment running Streamlit. "
        "To fix this, install it in the same environment (Anaconda base) and rerun the app."
    )

tab1, tab2 = st.tabs(["File Upload", "Live Webcam"])

with tab1:
    uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4"])
    if uploaded_file is not None and _ULTRALYTICS_OK:
        if uploaded_file.type.startswith('image'):
            image = Image.open(uploaded_file)
            # Convert PIL to BGR for consistent handling in annotate function
            img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            results = model(img_bgr)
            annotated = annotate_and_alert_v8(results, img_bgr)
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Detection Result")
        elif uploaded_file.type == 'video/mp4':
            tfile = open('temp.mp4', 'wb')
            tfile.write(uploaded_file.read())
            cap = cv2.VideoCapture('temp.mp4')
            stframe = st.empty()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                results = model(frame)
                annotated = annotate_and_alert_v8(results, frame)
                stframe.image(annotated, channels="BGR")
            cap.release()

with tab2:
    run = st.checkbox('Start Webcam')
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)
    while run and _ULTRALYTICS_OK:
        ret, frame = camera.read()
        if not ret:
            st.write("Failed to grab frame")
            break
        results = model(frame)
        annotated = annotate_and_alert_v8(results, frame)
        FRAME_WINDOW.image(annotated, channels="BGR")
    camera.release()

if not _ULTRALYTICS_OK:
    st.markdown(
        """
        How to install Ultralytics into the same environment Streamlit uses (Windows, Anaconda base):

        1) Close other terminals, then open Anaconda Prompt as Administrator.
        2) Run:
           - conda activate base
           - pip install ultralytics opencv-python pillow torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121
        3) Start the app using the same environment:
           - streamlit run "D:\\Saylani AI & DS\\python\\Deep Learning\\Cigarette and Vape Detection System\\app.py"

        If you prefer not to use Anaconda base, ensure your Streamlit terminal uses the Python where Ultralytics is installed.
        """
    )