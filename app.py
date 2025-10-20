import cv2
import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from ultralytics import YOLO

# Absolute path to your model
MODEL_PATH = r"D:\Projects\AI-ML\Ciggarrete and Vape Detection\best.pt"

# Load YOLO model once
model = YOLO(MODEL_PATH)

st.title("🚭 Cigarette & Vape Detection — Real-time Webcam")
st.markdown(
    "Allow webcam access and ensure 'best.pt' exists at the path above.\n"
    "The model will draw bounding boxes in real time."
)

# Custom transformer to process video frames
class YOLOv11Transformer(VideoTransformerBase):
    def __init__(self):
        self.model = model

    def transform(self, frame):
        # Convert WebRTC frame to OpenCV format (BGR)
        img = frame.to_ndarray(format="bgr24")

        # Run YOLOv11 inference
        results = self.model.predict(img, stream=False, verbose=False)

        # Draw detections on frame
        for r in results:
            boxes = r.boxes.xyxy  # [x1, y1, x2, y2, conf, class]
            confs = r.boxes.conf
            clss = r.boxes.cls

            for box, conf, cls in zip(boxes, confs, clss):
                x1, y1, x2, y2 = map(int, box[:4])
                label = f"{r.names[int(cls)]}: {conf:.2f}"
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Return annotated frame
        return img


# Start the Streamlit WebRTC component
webrtc_streamer(
    key="cig-detect",
    video_transformer_factory=YOLOv11Transformer,
    media_stream_constraints={"video": True, "audio": False},
)
