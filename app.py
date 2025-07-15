import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import io
from datetime import date, datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import base64
import csv
import os
import pandas as pd
from pathlib import Path
from scipy.interpolate import make_interp_spline

st.set_page_config(page_title="Retinitis Pigmentosa Detection", layout="wide")

# === MODEL PATHS ===
def get_base_path():
    try:
        return Path(__file__).parent
    except NameError:
        return Path.cwd()

BASE_PATH = get_base_path()
RETINA_MODEL_PATH = BASE_PATH / "retina_vs_nonretina.tflite"
DISEASE10_MODEL_PATH = BASE_PATH / "disease10_model.tflite"
RP_MODEL_PATH = BASE_PATH / "rp_detection_model.tflite"

# === LOAD MODELS WITH VALIDATION ===
@st.cache_resource
def load_tflite_model(path):
    try:
        if not path.exists():
            st.error(f"❌ Model file not found: {path}")
            return None
        if path.stat().st_size == 0:
            st.error(f"❌ Model file is empty: {path}")
            return None
        interpreter = tf.lite.Interpreter(model_path=str(path))
        interpreter.allocate_tensors()
        st.success(f"✅ Loaded: {path.name}")
        return interpreter
    except Exception as e:
        st.error(f"❌ Failed to load model `{path.name}`: {e}")
        return None

# === Load All Models ===
retina_model = load_tflite_model(RETINA_MODEL_PATH)
disease10_model = load_tflite_model(DISEASE10_MODEL_PATH)
rp_model = load_tflite_model(RP_MODEL_PATH)

# === DEBUG INFO ===
st.sidebar.markdown("### 🛠️ Debug Info")
st.sidebar.write(f"📁 Base path: `{BASE_PATH}`")
st.sidebar.write(f"✅ Retina model: `{RETINA_MODEL_PATH.exists()}`")
st.sidebar.write(f"✅ 10-class model: `{DISEASE10_MODEL_PATH.exists()}`")
st.sidebar.write(f"✅ RP model: `{RP_MODEL_PATH.exists()}`")

# === CLASS LABELS ===
class_names_rp = ['Healthy', 'Retinitis Pigmentosa']
class_names_10class = [
    "Healthy", "Retinitis Pigmentosa", "Diabetic Retinopathy", "Glaucoma",
    "Hypertensive Retinopathy", "Macular Hole", "Myopia",
    "Retinal Detachment", "Cataract", "Age-related Macular Degeneration"
]

confidence_threshold = 60.0

def tflite_predict(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])

# === STATE ===
st.title("👁️ Retinitis Pigmentosa Detection App")
if "scan_count" not in st.session_state:
    st.session_state.scan_count = 0
if "prediction_log" not in st.session_state:
    st.session_state.prediction_log = []

# === SIDEBAR: METRICS ===
st.sidebar.markdown("### 📊 App Statistics")
st.sidebar.markdown(f"**🧪 Scans:** {st.session_state.scan_count}")

st.sidebar.markdown("### 🧮 Metrics Summary")
st.sidebar.table(pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Sensitivity", "F1 Score"],
    "Value": ["96.67%", "96.7%", "97.0%", "97.8%"]
}))

# === FORM ===
with st.form("patient_form"):
    st.header("Patient Information")
    name = st.text_input("Name")
    dob = st.date_input("Date of Birth", min_value=date(1950, 1, 1), max_value=date.today())
    age = st.number_input("Age", min_value=0, max_value=120, step=1)
    blood_group = st.text_input("Blood Group")
    contact = st.text_input("Contact Number")
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    doctor = st.text_input("Doctor Name")
    hospital = st.text_input("Hospital Name")
    patient_id = st.text_input("Patient ID")
    image_file = st.file_uploader("Upload Retina Image", type=["jpg", "jpeg", "png"])
    submit = st.form_submit_button("Predict")

if submit:
    if image_file is None:
        st.error("❌ Please upload an image.")
    else:
        st.session_state.scan_count += 1
        image = Image.open(image_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        image_resized = image.resize((224, 224))
        img_array = np.expand_dims(np.array(image_resized) / 255.0, axis=0).astype(np.float32)

        image_diagnosis = "Unidentified"
        disease_diagnosis = "Not Applicable"
        predicted_disease = "Unknown"
        rp_confidence = 0.0

        # === Retina Check ===
        with st.spinner("🔍 Checking if retina..."):
            try:
                retina_output = tflite_predict(retina_model, img_array)[0][0]
                is_retina = retina_output < 0.5
                retina_confidence = (1 - retina_output) * 100
                if not is_retina:
                    st.warning(f"⚠️ Not a retina image (Confidence: {100 - retina_confidence:.2f}%)")
                else:
                    image_diagnosis = "Retina"
                    st.success(f"✅ Retina confirmed (Confidence: {retina_confidence:.2f}%)")
            except Exception as e:
                st.error(f"❌ Retina detection error: {e}")
                st.stop()

        # === 10-Class Disease Prediction ===
        if image_diagnosis == "Retina":
            with st.spinner("🔬 Predicting disease..."):
                try:
                    disease_output = tflite_predict(disease10_model, img_array)[0]
                    predicted_idx = np.argmax(disease_output)
                    predicted_disease = class_names_10class[predicted_idx]
                    disease_confidence = disease_output[predicted_idx] * 100
                    st.success(f"🧠 Disease: {predicted_disease} ({disease_confidence:.2f}%)")
                except Exception as e:
                    st.error(f"❌ 10-class model error: {e}")
                    st.stop()

            # === RP Confirmation ===
            if predicted_disease in ["Healthy", "Retinitis Pigmentosa"]:
                with st.spinner("🧠 Confirming RP..."):
                    try:
                        rp_output = tflite_predict(rp_model, img_array)[0][0]
                        prob_rp = float(rp_output)
                        prob_healthy = 1 - prob_rp
                        rp_confidence = max(prob_rp, prob_healthy) * 100
                        disease_diagnosis = class_names_rp[1] if prob_rp > 0.5 else class_names_rp[0]

                        if rp_confidence < confidence_threshold:
                            disease_diagnosis = "Uncertain"
                            st.warning("⚠️ RP confidence too low.")
                        else:
                            st.success(f"🔍 RP Verdict: {disease_diagnosis} ({rp_confidence:.2f}%)")
                    except Exception as e:
                        st.error(f"❌ RP model error: {e}")
                        st.stop()
            else:
                disease_diagnosis = "Unidentified"
                st.warning("🚫 RP model skipped for non-RP/Healthy cases.")

        # === Result Display ===
        st.markdown("## 🧾 Results")
        st.write(f"**Image Type:** {image_diagnosis}")
        st.write(f"**10-Class Prediction:** {predicted_disease}")
        st.write(f"**RP Status:** {disease_diagnosis}")
        if disease_diagnosis not in ["Uncertain", "Not Applicable"]:
            st.write(f"**Confidence:** {rp_confidence:.2f}%")

        # === Logging for Table ===
        st.session_state.prediction_log.append({
            "Image ID": len(st.session_state.prediction_log) + 1,
            "Predicted Disease": predicted_disease,
            "RP Verdict": disease_diagnosis,
            "Confidence": f"{rp_confidence:.2f}%"
        })

        # === PDF Report ===
        pdf_buffer = io.BytesIO()
        c = canvas.Canvas(pdf_buffer, pagesize=letter)
        c.setFont("Helvetica-Bold", 20)
        c.drawCentredString(300, 770, "Retina Disease Detection Report")
        y = 730
        c.setFont("Helvetica", 11)
        for line in [
            f"Name: {name}", f"DOB: {dob}", f"Age: {age}",
            f"Blood Group: {blood_group}", f"Contact: {contact}",
            f"Gender: {gender}", f"Doctor: {doctor}", f"Hospital: {hospital}", f"Patient ID: {patient_id}",
            f"Image Type: {image_diagnosis}", f"Predicted Disease: {predicted_disease}",
            f"RP Status: {disease_diagnosis}", f"Confidence: {rp_confidence:.2f}%"
        ]:
            c.drawString(50, y, line)
            y -= 15
        c.save()
        pdf_buffer.seek(0)
        b64_pdf = base64.b64encode(pdf_buffer.read()).decode('utf-8')
        href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="RP_Report_{name}_{datetime.now().strftime("%Y%m%d%H%M%S")}.pdf">📄 Download PDF</a>'
        st.markdown(href, unsafe_allow_html=True)

        # === Save to CSV ===
        with open("rp_report.csv", mode="a", newline="") as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(["Timestamp", "Name", "DOB", "Age", "Blood Group", "Contact", "Gender", "Doctor", "Hospital", "Patient ID", "Image Type", "Predicted Disease", "RP Status", "Confidence"])
            writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), name, dob, age, blood_group, contact, gender, doctor, hospital, patient_id, image_diagnosis, predicted_disease, disease_diagnosis, f"{rp_confidence:.2f}"])

        st.success("✅ Report saved successfully!")
