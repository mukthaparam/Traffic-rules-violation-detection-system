import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

# [Previous helper functions remain unchanged]
def load_yolo_model():
    try:
        model = YOLO("yolov8n.pt")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def check_bbox_overlap(bbox1, bbox2, threshold=0.5):
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    if x_right < x_left or y_bottom < y_top:
        return False
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    min_area = min(box1_area, box2_area)
    return intersection_area / min_area > threshold

def detect_violations(frame, model):
    results = model(frame, conf=0.3)
    detections = results[0].boxes
    violations = {
        "helmet_violation": [],
        "illegal_parking": [],
        "overloading": []
    }
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection.xyxy[0][:4].cpu().numpy())
        conf = float(detection.conf[0].cpu().numpy())
        class_id = int(detection.cls[0].cpu().numpy())
        color = (255, 255, 255)
        label = "Unknown"
        if class_id == 3:
            label = "Motorcycle"
            rider_count = 0
            helmet_count = 0
            for other_det in detections:
                other_class = int(other_det.cls[0].cpu().numpy())
                if other_class == 0:
                    other_box = map(int, other_det.xyxy[0][:4].cpu().numpy())
                    if check_bbox_overlap([x1, y1, x2, y2], list(other_box)):
                        rider_count += 1
                elif other_class == 44:
                    other_box = map(int, other_det.xyxy[0][:4].cpu().numpy())
                    if check_bbox_overlap([x1, y1, x2, y2], list(other_box)):
                        helmet_count += 1
            if rider_count > 2:
                violations["overloading"].append([x1, y1, x2, y2])
                color = (255, 255, 0)
                label = f"Overloading ({rider_count} riders)"
            if rider_count > 0 and helmet_count < rider_count:
                violations["helmet_violation"].append([x1, y1, x2, y2])
                label = f"No Helmet ({rider_count - helmet_count} riders)"
                color = (0, 0, 255)
        elif class_id == 7:
            label = "Truck"
            load_height = y2 - y1
            truck_length = x2 - x1
            if (load_height / truck_length) > 1.5:
                violations["overloading"].append([x1, y1, x2, y2])
                color = (255, 255, 0)
                label = "Truck Overloading"
        elif class_id in [2, 7]:
            label = "Vehicle"
            frame_height = frame.shape[0]
            if y2 > (frame_height * 0.7):
                color = (255, 0, 0)
                label = "Illegal Parking"
                violations["illegal_parking"].append([x1, y1, x2, y2])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame, violations

st.set_page_config(page_title="Traffic Violation Detection", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
        .stApp {
            background-color: #000000;
        }
        .title-container {
            background: rgba(30, 41, 59, 0.5);
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .title-text {
            font-size: 1.5em !important;
            font-weight: 700 !important;
            color: white !important;
            text-align: center;
            margin-bottom: 0.3em !important;
        }
        .subtitle-text {
            font-size: 0.8em !important;
            text-align: center;
            color: #94a3b8 !important;
        }
        .info-card {
            background: rgba(30, 41, 59, 0.5);
            border-radius: 8px;
            padding: 0.75rem;
            margin: 0.5rem 0;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .upload-container {
            background: rgba(30, 41, 59, 0.5);
            border-radius: 8px;
            padding: 0.75rem;
            text-align: center;
            border: 2px dashed rgba(148, 163, 184, 0.2);
            margin: 1rem 0;
        }
        .results-container {
            background: rgba(30, 41, 59, 0.5);
            border-radius: 8px;
            padding: 0.75rem;
            margin-top: 1rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
            max-width: 800px;
            margin-left: auto;
            margin-right: auto;
        }
        .stats-box {
            background: rgba(51, 65, 85, 0.5);
            border-radius: 8px;
            padding: 0.5rem;
            margin: 0.25rem 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .stats-label {
            font-size: 0.8em;
            color: #94a3b8;
        }
        .stats-value {
            font-size: 0.9em;
            font-weight: bold;
            color: #60a5fa;
        }
        .stButton>button {
            background: linear-gradient(45deg, #3b82f6, #2563eb);
            color: white !important;
            border: none !important;
            padding: 0.5rem 1.5rem !important;
            border-radius: 6px !important;
            font-weight: 600 !important;
            transition: all 0.3s ease;
            width: auto !important;
        }
        .stButton>button:hover {
            background: linear-gradient(45deg, #2563eb, #1d4ed8);
            transform: translateY(-2px);
        }
        .alert {
            background: rgba(234, 179, 8, 0.1);
            border: 1px solid rgba(234, 179, 8, 0.3);
            border-radius: 6px;
            padding: 0.75rem;
            margin: 0.75rem 0;
            color: #fef08a;
        }
        img {
            max-width: 800px !important;
            margin: 0 auto !important;
            display: block !important;
        }
        h1, h2, h3, h4, h5, h6 {
            color: white !important;
            font-size: 1em !important;
        }
        .upload-text {
            color: #94a3b8 !important;
            font-size: 0.8em !important;
            margin-bottom: 0.5rem !important;
        }
        .stImage {
            border-radius: 8px;
            overflow: hidden;
        }
        p, li {
            font-size: 0.8em !important;
            color: #94a3b8 !important;
        }
    </style>
""", unsafe_allow_html=True)

def main():
    st.markdown("""
        <div class="title-container">
            <div class="title-text">🚦 Traffic Violation Detection</div>
            <div class="subtitle-text">AI-powered traffic monitoring system</div>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("""
            <div class="info-card">
                <h3>🎯 Detectable Violations</h3>
                <ul style="color: #94a3b8; list-style-type: none; padding-left: 0;">
                    <li>🛑 Illegal Parking</li>
                    <li>⛑️ No Helmet</li>
                    <li>📦 Overloading</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class="info-card">
                <h3>💡 System Overview</h3>
                <p style="color: #94a3b8;">Our advanced AI system processes traffic footage in real-time to detect and document violations, ensuring safer roads for everyone.</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="upload-container">', unsafe_allow_html=True)
    st.markdown('<p class="upload-text">Drop your traffic scene image here</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file:
        with st.spinner("🔍 Analyzing image..."):
            model = load_yolo_model()
            if model:
                image = Image.open(uploaded_file)
                image_np = np.array(image)
                if len(image_np.shape) == 2:
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
                elif image_np.shape[2] == 4:
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
                
                processed_image, violations = detect_violations(image_np, model)
                st.markdown('<div class="results-container">', unsafe_allow_html=True)
                st.image(processed_image, caption="Analysis Results", use_column_width=True)
                
                total_violations = sum(len(v) for v in violations.values())
                if total_violations > 0:
                    st.markdown("""
                        <div class="alert">
                            ⚠️ Multiple violations detected. Review the analysis below.
                        </div>
                    """, unsafe_allow_html=True)
                
                cols = st.columns(4)
                stats = [
                    {"label": "Total Violations", "value": total_violations},
                    {"label": "No Helmet", "value": len(violations['helmet_violation'])},
                    {"label": "Illegal Parking", "value": len(violations['illegal_parking'])},
                    {"label": "Overloading", "value": len(violations['overloading'])}
                ]
                
                for idx, stat in enumerate(stats):
                    with cols[idx]:
                        st.markdown(f"""
                            <div class="stats-box">
                                <span class="stats-label">{stat['label']}</span>
                                <span class="stats-value">{stat['value']}</span>
                            </div>
                        """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()