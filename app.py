import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import requests
import smtplib
from email.mime.text import MIMEText
import csv
from datetime import datetime
import gdown # type: ignore
import os

MODEL_PATH = "best_soil_model.pth"

if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=1wYlKKDvqh8U_41-N81ZUYy9riy8LcMGk"
    gdown.download(url, MODEL_PATH, quiet=False)

# -------------------------

# CONFIG
# -------------------------
st.set_page_config(page_title="Soil Classifier", layout="centered")

soil_labels = ['alluvial', 'black', 'clay', 'red', 'yellow']
model_accuracy = "97%"

# -------------------------
# LOAD MODEL
# -------------------------
@st.cache_resource
def load_model():
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(soil_labels))
    model.load_state_dict(torch.load("best_soil_model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# -------------------------
# TRANSFORM
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

device = torch.device("cpu")

# -------------------------
# WEATHER
# -------------------------
def get_weather(city):
    api_key = "e600cfd7f0d948f281183753262402"
    url = f"https://api.weatherapi.com/v1/current.json?key={api_key}&q={city}"

    try:
        data = requests.get(url).json()
        if "error" in data:
            return None, None
        return data["current"]["humidity"], data["current"]["temp_c"]
    except:
        return None, None

# -------------------------
# GRAIN SIZE
# -------------------------
def grain_size_estimate(soil_type):
    if soil_type == "red":
        return "Coarse to medium particle size with good drainage."
    elif soil_type == "alluvial":
        return "Mixed particle size including sand and silt."
    elif soil_type == "clay":
        return "Very fine particle size with low permeability."
    elif soil_type == "black":
        return "Fine particle size with high moisture retention."

# -------------------------
# EMAIL
# -------------------------
def send_email_report(soil_type, humidity, quality, grain_size, risk):
    sender = "aagamjain816@gmail.com"
    receiver = "aagamjain516@gmail.com"
    password = "eqtbmlqavgcpzfrv"

    body = f"""
Soil Type: {soil_type}
Humidity: {humidity}%

Quality: {quality}
Grain Size: {grain_size}
Risk: {risk}
"""

    msg = MIMEText(body)
    msg["Subject"] = "Soil Monitoring Report"
    msg["From"] = sender
    msg["To"] = receiver

    try:
        server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
        server.login(sender, password)
        server.sendmail(sender, receiver, msg.as_string())
        server.quit()
        st.success("Email sent successfully")
    except:
        st.error("Email sending failed")

# -------------------------
# CIVIL ANALYSIS
# -------------------------
def civil_analysis(soil, humidity):
    if soil == "clay":
        return "75–150 kN/m²", "High settlement", "Pile/raft foundation"
    elif soil == "black":
        return "50–100 kN/m²", "Very high settlement", "Deep foundation"
    elif soil == "alluvial":
        return "100–200 kN/m²", "Moderate settlement", "Raft footing"
    elif soil == "red":
        return "150–300 kN/m²", "Low settlement", "Shallow foundation"

# -------------------------
# QUALITY
# -------------------------
def soil_quality_grade(soil_type, humidity):
    grade = {
        "red": "Grade A",
        "alluvial": "Grade B",
        "clay": "Grade C",
        "black": "Grade C"
    }[soil_type]

    if humidity > 80:
        grade += " (High moisture)"

    return grade

# -------------------------
# RISK
# -------------------------
def risk_alert(settlement, humidity):
    if "very high" in settlement.lower():
        return "HIGH RISK"
    elif humidity > 80:
        return "MODERATE RISK"
    return "LOW RISK"

# -------------------------
# LOGGING
# -------------------------
def log_data(soil_type, humidity, risk):
    with open("iot_soil_log.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now(), soil_type, humidity, risk])

# -------------------------
# UI
# -------------------------
st.title("AI-Based Soil Classification")

st.sidebar.title("Model Info")
st.sidebar.write(f"Accuracy: {model_accuracy}")

uploaded_file = st.file_uploader("Upload Soil Image", type=["jpg", "png", "jpeg"])
city = st.text_input("Enter City")

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image")

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, 1)

    soil_type = soil_labels[predicted.item()]
    confidence = confidence.item() * 100

    st.subheader(f"Soil Type: {soil_type.upper()}")
    st.write(f"Confidence: {confidence:.2f}%")

    if city:
        humidity, temp = get_weather(city)

        if humidity:
            bearing, settlement, foundation = civil_analysis(soil_type, humidity)
            quality = soil_quality_grade(soil_type, humidity)
            risk = risk_alert(settlement, humidity)
            grain = grain_size_estimate(soil_type)

            st.write(f"Temperature: {temp} C | Humidity: {humidity}%")
            st.write("Quality:", quality)
            st.write("Bearing:", bearing)
            st.write("Settlement:", settlement)
            st.write("Foundation:", foundation)
            st.write("Risk:", risk)
            st.write("Grain Size:", grain)

            if st.button("Send Email Report"):
                send_email_report(soil_type, humidity, quality, grain, risk)

            log_data(soil_type, humidity, risk)

        else:
            st.error("City not found")