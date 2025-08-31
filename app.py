import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import requests
from cryptography.fernet import Fernet
from sklearn.linear_model import LogisticRegression
from ultralytics import YOLO


# ==============================
# Utility Functions
# ==============================
def encrypt_message(message, key):
    f = Fernet(key)
    return f.encrypt(message.encode()).decode()

def decrypt_message(token, key):
    f = Fernet(key)
    return f.decrypt(token.encode()).decode()

# Load YOLOv8 model (pretrained on COCO dataset)
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # nano version for speed

# Fetch real-time weather data
def get_weather_data(lat, lon, api_key):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            weather = {
                "temperature": data["main"]["temp"],
                "visibility": data.get("visibility", 10000)/1000,
                "wind_speed": data["wind"]["speed"]
            }
            return weather
        else:
            return None
    except:
        return None

# Fetch terrain difficulty from Google Elevation API
def get_terrain_difficulty(lat, lon, api_key):
    try:
        url = f"https://maps.googleapis.com/maps/api/elevation/json?locations={lat},{lon}&key={api_key}"
        response = requests.get(url)
        data = response.json()
        if 'results' in data and len(data['results']) > 0:
            elevation = data['results'][0]['elevation']
            return min(max(int(elevation/1000*10), 1), 10)
    except:
        return 5  # default
    return 5

# ==============================
# Streamlit App
# ==============================
st.set_page_config(page_title="Sentinel-X", layout="wide")
st.title("🛰️ Sentinel-X: AI Powered Unified Defense Dashboard")

menu = ["Home", "Surveillance AI", "Secure Comms", "Threat Prediction", "Soldier Support"]
choice = st.sidebar.radio("Navigate", menu)

# ==============================
# HOME
# ==============================
if choice == "Home":
    st.header("Welcome to Sentinel-X")
    st.write("This is a prototype digital defense ecosystem built for Hackathons.")
    st.markdown("""
    **Modules Available:**
    - Surveillance AI: Detect humans/objects from video feeds.
    - Secure Comms: End-to-end encrypted soldier communication.
    - Threat Prediction: Real-time weather + incident/terrain → threat assessment.
    - Soldier Support: Real-time vitals and SOS alert system.
    """)

# ==============================
# Surveillance AI
# ==============================
elif choice == "Surveillance AI":
    st.header("🔍 Border Surveillance AI")
    uploaded_file = st.file_uploader("Upload a border surveillance video", type=["mp4", "avi"])

    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        model = load_model()
        st.info("Processing video with YOLOv8...")

        cap = cv2.VideoCapture(tfile.name)
        frame_placeholder = st.empty()
        stats_placeholder = st.empty()
        summary_counts = {}

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, verbose=False)
            annotated = results[0].plot()

            # Count objects
            class_names = results[0].names
            boxes = results[0].boxes
            frame_counts = {}
            for c in boxes.cls.cpu().numpy().astype(int):
                label = class_names[c]
                frame_counts[label] = frame_counts.get(label, 0) + 1
                summary_counts[label] = summary_counts.get(label, 0) + 1

            frame_placeholder.image(annotated, channels="BGR")
            if frame_counts:
                stats_placeholder.markdown(
                    "**Detected Objects (Live):**<br>" +
                    "<br>".join([f"- {obj}: {count}" for obj, count in frame_counts.items()]),
                    unsafe_allow_html=True
                )

        cap.release()
        st.success("Processing Complete 🚀")

        if summary_counts:
            st.subheader("📊 Overall Detection Summary")
            summary_df = pd.DataFrame(list(summary_counts.items()), columns=["Object", "Count"])
            st.table(summary_df)

# ==============================
# Secure Communication
# ==============================
elif choice == "Secure Comms":
    st.header("🔒 Secure Soldier Communication")
    if "key" not in st.session_state:
        st.session_state["key"] = Fernet.generate_key()
        st.session_state["messages"] = []

    message = st.text_input("Enter your message")
    if st.button("Send") and message:
        encrypted = encrypt_message(message, st.session_state["key"])
        st.session_state["messages"].append((message, encrypted))

    st.subheader("Chat History")
    for original, encrypted in st.session_state["messages"]:
        st.write(f"🧑 Soldier: {original}")
        st.code(f"Encrypted: {encrypted}")

# ==============================
# Threat Prediction
# ==============================
elif choice == "Threat Prediction":
    st.header("⚠️ Predictive Threat Assessment – Ladakh Border")

    # Default Leh, Ladakh coordinates
    lat = 34.1526
    lon = 77.5770

    api_key_weather = st.text_input("Enter OpenWeatherMap API Key", type="password")
    api_key_google = st.text_input("Enter Google Maps Elevation API Key", type="password")
    gtd_file = st.file_uploader("Upload Global Terrorism Dataset CSV", type="csv")

    if st.button("Fetch Data & Predict Threat"):
        # Weather
        weather = get_weather_data(lat, lon, api_key_weather)
        if weather:
            st.write("### 🌦️ Current Weather")
            st.metric("Temperature (°C)", weather["temperature"])
            st.metric("Visibility (km)", weather["visibility"])
            st.metric("Wind Speed (m/s)", weather["wind_speed"])
        else:
            st.warning("Weather API failed. Using fallback values.")
            weather = {"temperature": 15, "visibility": 8, "wind_speed": 5}

        # Incident count
        incident_count = 0
        if gtd_file:
            df = pd.read_csv(gtd_file)
            df = df[(df['country_txt'] == 'India') & (df['iyear'] >= 2015)]
            df = df[(df['latitude'].between(lat-0.5, lat+0.5)) & 
                    (df['longitude'].between(lon-0.5, lon+0.5))]
            incident_count = len(df)
        else:
            incident_count = np.random.randint(0, 5)
        st.write(f"### 🗂️ Incident Count: {incident_count}")

        # Terrain difficulty
        terrain_difficulty = get_terrain_difficulty(lat, lon, api_key_google)
        st.write(f"### 🗻 Terrain Difficulty: {terrain_difficulty}")

        # Logistic Regression Model (synthetic)
        X_train = pd.DataFrame({
            "temperature": np.random.randint(-10, 50, 50),
            "visibility": np.random.randint(1, 10, 50),
            "wind_speed": np.random.randint(0, 20, 50),
            "incident_count": np.random.randint(0, 10, 50),
            "terrain_difficulty": np.random.randint(1, 10, 50)
        })
        y_train = np.random.randint(0, 2, 50)

        model = LogisticRegression()
        model.fit(X_train, y_train)

        X_new = pd.DataFrame([[
            weather["temperature"],
            weather["visibility"],
            weather["wind_speed"],
            incident_count,
            terrain_difficulty
        ]], columns=X_train.columns)

        pred = model.predict(X_new)[0]
        prob = model.predict_proba(X_new)[0][1]

        st.write("### ⚠️ Threat Assessment")
        st.metric("Predicted Threat", "High" if pred==1 else "Low")
        st.progress(float(prob))

# ==============================
# Soldier Support
# ==============================
elif choice == "Soldier Support":
    st.header("🪖 Soldier Health Monitoring")

    if st.button("Start Monitoring"):
        hr = np.random.randint(50, 120)
        oxy = np.random.randint(85, 100)

        col1, col2 = st.columns(2)
        col1.metric("Heart Rate", f"{hr} bpm")
        col2.metric("Oxygen Level", f"{oxy}%")

        if hr < 60 or oxy < 90:
            st.error("⚠️ SOS Alert! Soldier in distress")
        else:
            st.success("✅ Soldier Stable")
