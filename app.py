import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
import io
from datetime import date, datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import base64
import csv
import os
import gdown

st.set_page_config(page_title="Retinitis Pigmentosa Detection", layout="wide")

@st.cache_resource
def download_models():
    # Google Drive file IDs
    retina_drive_id = "1gc6UKCOY-eo5C9-9QbIT4Jo-0RRucXKl"  # retina_vs_nonretina.h5
    rp_drive_id = "1eZgrVcdMBtT3i7lVtZS7KwzBqordBL00"       # rp_detection_model.h5

    retina_model_path = "retina_vs_nonretina.h5"
    rp_model_path = "rp_detection_model.h5"

    if not os.path.exists(retina_model_path):
        st.info("Downloading retina_vs_nonretina.h5 model...")
        gdown.download(f"https://drive.google.com/uc?id={retina_drive_id}&export=download", retina_model_path, quiet=False)
    else:
        st.success(f"{retina_model_path} already downloaded.")

    if not os.path.exists(rp_model_path):
        st.info("Downloading rp_detection_model.h5 model...")
        gdown.download(f"https://drive.google.com/uc?id={rp_drive_id}&export=download", rp_model_path, quiet=False)
    else:
        st.success(f"{rp_model_path} already downloaded.")

    # Check files
    st.write("Files in current directory:", os.listdir())

    return retina_model_path, rp_model_path

@st.cache_resource
def load_model_cached(path):
    try:
        model = load_model(path)
        st.success(f"Model loaded successfully: {path}")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model from {path}: {e}")
        return None

# Download and load models
retina_model_path, rp_model_path = download_models()
retina_model = load_model_cached(retina_model_path)
rp_model = load_model_cached(rp_model_path)

class_names_rp = ['Healthy', 'Retinitis Pigmentosa']
confidence_threshold = 60.0

st.title("👁️ Retinitis Pigmentosa Detection App")

if "scan_count" not in st.session_state:
    st.session_state.scan_count = 0

st.sidebar.markdown("### 📊 App Statistics")
st.sidebar.markdown(f"**🧪 Images Scanned:** {st.session_state.scan_count}")

# Accuracy graph (dummy data)
st.sidebar.markdown("### 📈 Model Accuracy Graph")
fig_acc, ax_acc = plt.subplots()
epochs = list(range(1, 11))
train_acc = [0.967 + i * 0.001 for i in range(10)]
val_acc = [0.966 + i * 0.001 for i in range(10)]
ax_acc.plot(epochs, train_acc, label='Training Accuracy')
ax_acc.plot(epochs, val_acc, label='Validation Accuracy')
ax_acc.set_ylim(0.95, 1.0)
ax_acc.set_xlabel('Epoch')
ax_acc.set_ylabel('Accuracy')
ax_acc.set_title('Model Accuracy Over Epochs')
ax_acc.legend()
st.sidebar.pyplot(fig_acc)

with st.form("patient_form"):
    st.header("Patient Information")
    name = st.text_input("Name")
    dob = st.date_input("Date of Birth", min_value=date(1950,1,1), max_value=date.today())
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
    elif retina_model is None or rp_model is None:
        st.error("❌ Models not loaded. Please check your model files and download links.")
    else:
        st.session_state.scan_count += 1
        image = Image.open(image_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        retina_img = image.resize((224, 224))
        retina_arr = np.expand_dims(np.array(retina_img) / 255.0, axis=0)

        image_diagnosis = "Unidentified"
        disease_diagnosis = "Not Applicable"
        retina_confidence = 0.0
        rp_confidence = 0.0
        prob_rp, prob_healthy = 0.0, 0.0

        with st.spinner("🔍 Checking if the image is a retina..."):
            try:
                retina_pred = retina_model.predict(retina_arr)[0][0]
                retina_confidence = (1 - retina_pred) * 100
                is_retina = retina_pred < 0.5
                if not is_retina:
                    image_diagnosis = "Unidentified"
                    st.warning(f"⚠️ Not a retina image (Confidence: {100 - retina_confidence:.2f}%)")
                else:
                    image_diagnosis = "Retina"
                    st.success(f"✅ Retina image confirmed (Confidence: {retina_confidence:.2f}%)")
            except Exception as e:
                st.error(f"Retina classification error: {e}")

        if image_diagnosis == "Retina":
            rp_img = image.resize((224, 224))
            rp_arr = np.expand_dims(np.array(rp_img) / 255.0, axis=0)

            with st.spinner("🧠 Detecting Retinitis Pigmentosa..."):
                try:
                    rp_pred = rp_model.predict(rp_arr)[0][0]
                    prob_rp = float(rp_pred)
                    prob_healthy = 1 - prob_rp
                    rp_confidence = max(prob_rp, prob_healthy) * 100
                    if rp_confidence < confidence_threshold:
                        disease_diagnosis = "Uncertain"
                        st.warning("⚠️ Prediction confidence is low.")
                    else:
                        disease_diagnosis = class_names_rp[1] if prob_rp > 0.5 else class_names_rp[0]
                        st.success(f"🧠 Disease Prediction: {disease_diagnosis} (Confidence: {rp_confidence:.2f}%)")
                except Exception as e:
                    st.error(f"RP prediction error: {e}")

        st.markdown("## 🧾 Prediction Results")
        st.write(f"**Image Type Diagnosis:** {image_diagnosis}")
        st.write(f"**Disease Status:** {disease_diagnosis}")
        if disease_diagnosis not in ["Uncertain", "Not Applicable"]:
            st.write(f"**Confidence:** {rp_confidence:.2f}%")

        if image_diagnosis == "Retina" and disease_diagnosis not in ["Uncertain", "Not Applicable"]:
            fig, ax = plt.subplots()
            ax.pie([prob_healthy, prob_rp], labels=class_names_rp, colors=["green", "red"], autopct="%1.1f%%")
            ax.axis('equal')
            st.pyplot(fig)

        # Generate PDF
        pdf_buffer = io.BytesIO()
        c = canvas.Canvas(pdf_buffer, pagesize=letter)
        c.setFont("Helvetica-Bold", 20)
        c.drawCentredString(300, 770, "Retinitis Pigmentosa Report")
        c.line(40, 760, 570, 760)
        y = 730
        c.setFont("Helvetica-Bold", 12)
        c.drawString(40, y, "Patient Info")
        c.setFont("Helvetica", 11)
        y -= 20
        info = [
            f"Name: {name}", f"Date of Birth: {dob}", f"Age: {age}",
            f"Blood Group: {blood_group}", f"Contact: {contact}",
            f"Gender: {gender}", f"Doctor: {doctor}",
            f"Hospital: {hospital}", f"Patient ID: {patient_id}",
        ]
        for line in info:
            y -= 15
            c.drawString(50, y, line)
        y -= 20
        c.line(40, y, 570, y)
        y -= 20
        c.setFont("Helvetica-Bold", 12)
        c.drawString(40, y, "Prediction Results")
        c.setFont("Helvetica", 11)
        y -= 20
        c.drawString(50, y, f"Image Diagnosis: {image_diagnosis}")
        y -= 15
        c.drawString(50, y, f"Disease Status: {disease_diagnosis}")
        y -= 15
        if disease_diagnosis not in ["Uncertain", "Not Applicable"]:
            c.drawString(50, y, f"Confidence: {rp_confidence:.2f}%")
        c.save()
        pdf_buffer.seek(0)

        b64_pdf = base64.b64encode(pdf_buffer.read()).decode('utf-8')
        href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="RP_Report_{name}_{datetime.now().strftime("%Y%m%d%H%M%S")}.pdf">📄 Download Report PDF</a>'
        st.markdown(href, unsafe_allow_html=True)

        # Save to CSV
        csv_file = "rp_report.csv"
        file_exists = os.path.isfile(csv_file)
        with open(csv_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "Timestamp", "Name", "DOB", "Age", "Blood Group", "Contact",
                    "Gender", "Doctor", "Hospital", "Patient ID",
                    "Image Diagnosis", "Disease Status", "Confidence"
                ])
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"), name, dob, age, blood_group, contact,
                gender, doctor, hospital, patient_id,
                image_diagnosis, disease_diagnosis, f"{rp_confidence:.2f}"
            ])

        st.success("✅ Report saved successfully!")

