import cv2
import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

# --- PAGE CONFIG ---
st.set_page_config(page_title="Twite AI - Face Analytics", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-title {text-align: center; font-size: 2em; color: #4CAF50;}
    .metric-box {border: 2px solid #4CAF50; padding: 10px; border-radius: 10px; text-align: center;}
    .sidebar .stCheckbox, .sidebar .stSlider {margin-bottom: 20px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">👤 Real-Time Face & Gender Analytics</h1>', unsafe_allow_html=True)
st.markdown("Developed for Twite AI Technical Task")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Settings")
run = st.sidebar.checkbox('Start System', value=True)
conf_threshold = st.sidebar.slider("Detection Confidence", 1.1, 1.5, 1.3)
reset = st.sidebar.button("Reset Counts")

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    face_net = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gender_proto = "gender_deploy.prototxt"
    gender_model = "gender_net.caffemodel"
    gender_net = cv2.dnn.readNet(gender_model, gender_proto)
    return face_net, gender_net

face_cascade, gender_net = load_models()

# Constants
GENDER_LIST = ['Male', 'Female']
MODEL_MEAN_VALUES = (78.42, 87.76, 114.89)

# Initialize session state for counts
if 'total_faces' not in st.session_state:
    st.session_state.total_faces = 0
if 'male_count' not in st.session_state:
    st.session_state.male_count = 0
if 'female_count' not in st.session_state:
    st.session_state.female_count = 0
if 'gender_history' not in st.session_state:
    st.session_state.gender_history = []

if reset:
    st.session_state.total_faces = 0
    st.session_state.male_count = 0
    st.session_state.female_count = 0
    st.session_state.gender_history = []

# --- METRICS SECTION ---
st.subheader("📊 Live Metrics")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.metric("Total Faces Detected", st.session_state.total_faces)
    st.markdown('</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.metric("Males", st.session_state.male_count)
    st.markdown('</div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.metric("Females", st.session_state.female_count)
    st.markdown('</div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.metric("Last Detected", st.session_state.gender_history[-1] if st.session_state.gender_history else "None")
    st.markdown('</div>', unsafe_allow_html=True)

# --- VIDEO FEED ---
st.subheader("🎥 Live Video Feed")
frame_window = st.image([])

# --- ANALYTICS CHART ---
st.subheader("📈 Gender Distribution")
chart_placeholder = st.empty()

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to access webcam.")
        break

    # 1. PRE-PROCESSING
    display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 2. DETECTION
    faces = face_cascade.detectMultiScale(gray, conf_threshold, 5)
    
    # Update counts
    st.session_state.total_faces += len(faces)
    
    current_genders = []

    for (x, y, w, h) in faces:
        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

        # 3. GENDER INFERENCE
        face_crop = frame[y:y+h, x:x+w].copy()
        blob = cv2.dnn.blobFromImage(face_crop, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        gender_net.setInput(blob)
        preds = gender_net.forward()
        gender = GENDER_LIST[preds[0].argmax()]
        current_genders.append(gender)
        
        if gender == 'Male':
            st.session_state.male_count += 1
        else:
            st.session_state.female_count += 1

        st.session_state.gender_history.append(gender)

        # UI Overlay
        label = f"{gender}"
        cv2.putText(display_frame, label, (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Update chart
    if st.session_state.gender_history:
        gender_counts = pd.Series(st.session_state.gender_history).value_counts()
        fig, ax = plt.subplots()
        gender_counts.plot(kind='bar', ax=ax, color=['blue', 'pink'])
        ax.set_title("Gender Distribution")
        ax.set_ylabel("Count")
        chart_placeholder.pyplot(fig)

    # 4. RENDER TO WEB UI
    frame_window.image(display_frame, channels="RGB")

cap.release()
