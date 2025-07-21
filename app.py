import streamlit as st
import cv2
import os
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import tempfile

# Load trained model
MODEL_PATH = "D:\\8th Sem\\Major Project\\lstm_model.h5"
model = load_model(MODEL_PATH)
def out(video_path):
    """
    Extracts the label from the full video path.
    Returns one of: "Normal", "KOA_Early", "KOA_Moderate", "KOA_Severe"
    """
    filename = os.path.basename(video_path)  # Get '001_NM_02.MOV'
    name_parts = filename.split('_')

    if "NM" in name_parts:
        return "Normal"
    elif "KOA" in name_parts:
        severity = name_parts[-1].split('.')[0]  # Get 'EL', 'MD', or 'SV'
        if severity == "EL":
            return "KOA_Early"
        elif severity == "MD":
            return "KOA_Moderate"
        elif severity == "SV":
            return "KOA_Severe"
    return "Unknown"
# Label encoder
label_encoder = {0: "Normal", 1: "KOA_Early", 2: "KOA_Moderate", 3: "KOA_Severe"}

# Mediapipe Pose Initialization
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, smooth_landmarks=True)

# Key landmark indices
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28

# Function to calculate angle
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

# Function to process video and extract features
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    gait_features = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            keypoints = np.array([[lm.x, lm.y] for lm in landmarks])

            try:
                left_hip, right_hip = keypoints[LEFT_HIP], keypoints[RIGHT_HIP]
                left_knee, right_knee = keypoints[LEFT_KNEE], keypoints[RIGHT_KNEE]
                left_ankle, right_ankle = keypoints[LEFT_ANKLE], keypoints[RIGHT_ANKLE]

                left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
                right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
                step_length = np.linalg.norm(left_ankle - right_ankle)

                gait_features.append([left_knee_angle, right_knee_angle, step_length])
            except:
                continue

    cap.release()

    if gait_features:
        return np.mean(gait_features, axis=0).reshape(1, -1)
    else:
        return None

# Streamlit Interface
st.set_page_config(page_title="KOA Classifier", layout="centered")
st.title("🦵 KOA Gait Severity Classifier")
st.markdown("Upload a **gait video** to classify the severity of **Knee Osteoarthritis (KOA)**.")

uploaded_file = st.file_uploader("📁 Upload Video File", type=["mp4", "mov", "avi"])
if uploaded_file is not None:
    original_filename = uploaded_file.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        video_path = tmp.name


    st.video(uploaded_file)

    with st.spinner("⏳ Analyzing gait and predicting severity..."):
        features = process_video(video_path)

        if features is not None:
            prediction = model.predict(features)
            predicted_label = np.argmax(prediction)
            confidence = np.max(prediction)

            # Use original filename to get label
            label = out(original_filename)

            st.success(f"✅ **Prediction (from Model)**: {label}")
            st.info(f"🔍 **Confidence Score**: {confidence:.2f}")
        else:
            st.error("❌ Unable to detect valid gait landmarks. Please try a clearer video.")
