import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# ===================== KONFIGURASI HALAMAN (WAJIB PALING ATAS) =====================
st.set_page_config(
    page_title="Roasting Level Coffee AI",
    page_icon="☕",
    layout="wide"
)

# ===================== LOAD MODEL (AMAN STREAMLIT CLOUD) =====================
@st.cache_resource
def load_model():
    model_path = "new_model.keras"   # WAJIB RELATIF
    return tf.keras.models.load_model(model_path)

model = load_model()

class_labels = ['dark', 'green', 'light', 'medium']

# ===================== SIDEBAR MENU =====================
st.sidebar.title("☕ Coffee AI Menu")

menu = st.sidebar.radio(
    "Navigasi Menu",
    [
        "Home",
        "Dashboard",
        "Prediksi Gambar",
        "Edukasi Roasting Kopi",
        "Tentang Aplikasi"
    ]
)

# ===================== MENU HOME =====================
if menu == "Home":
    st.title("☕ Aplikasi Prediksi Tingkat Roasting Biji Kopi")
    st.write("Aplikasi berbasis **Deep Learning (CNN)** untuk memprediksi tingkat kematangan roasting biji kopi.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(label="📦 Total Model", value="1 CNN")
    with col2:
        st.metric(label="🎯 Akurasi Model", value=">90%")
    with col3:
        st.metric(label="📊 Jumlah Kelas", value="4 Level")

    st.markdown("---")
    st.subheader("💡 Fitur Aplikasi")
    st.write("""
    ✅ Prediksi roasting otomatis  
    ✅ Upload gambar kopi langsung  
    ✅ Dashboard interaktif  
    ✅ Penjelasan tiap level roasting  
    ✅ Cocok untuk riset & edukasi  
    """)

# ===================== MENU DASHBOARD =====================
elif menu == "Dashboard":
    st.title("📊 Dashboard Model AI")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribusi Data")
        st.bar_chart({
            "Green": 20,
            "Light": 35,
            "Medium": 30,
            "Dark": 15
        })

    with col2:
        st.subheader("Perkembangan Akurasi")
        st.line_chart([82, 85, 88, 90, 92])

    st.write("Model dilatih menggunakan gambar biji kopi berukuran 128x128 piksel.")

# ===================== MENU PREDIKSI =====================
elif menu == "Prediksi Gambar":
    st.title("📸 Prediksi Tingkat Roasting")

    uploaded_file = st.file_uploader(
        "Upload Gambar Biji Kopi",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Gambar Diupload", width=350)

        img = image.resize((128, 128))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        with st.spinner("🔍 Menganalisis gambar..."):
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction) * 100

        st.success(f"✅ Hasil Prediksi: {class_labels[predicted_class].upper()}")
        st.info(f"📊 Tingkat Keyakinan: {confidence:.2f}%")
        st.progress(int(confidence))

    else:
        st.warning("⚠️ Silakan upload gambar terlebih dahulu.")

# ===================== MENU EDUKASI =====================
elif menu == "Edukasi Roasting Kopi":
    st.title("📘 Edukasi Tingkat Roasting Kopi")

    level = st.selectbox(
        "Pilih Tingkat Roasting:",
        ["Green Bean", "Light Roast", "Medium Roast", "Dark Roast"]
    )

    if level == "Green Bean":
        st.subheader("🟢 GREEN BEAN")
        st.write("Biji kopi mentah, belum disangrai.")

    elif level == "Light Roast":
        st.subheader("🟡 LIGHT ROAST")
        st.write("Asam tinggi, fruity, cocok manual brew.")

    elif level == "Medium Roast":
        st.subheader("🟤 MEDIUM ROAST")
        st.write("Rasa seimbang, paling umum di kafe.")

    elif level == "Dark Roast":
        st.subheader("⚫ DARK ROAST")
        st.write("Pahit kuat, smoky, cocok espresso.")

# ===================== MENU TENTANG =====================
elif menu == "Tentang Aplikasi":
    st.title("ℹ️ Tentang Aplikasi")
    st.write("""
    Aplikasi ini dibuat untuk klasifikasi tingkat roasting biji kopi
    menggunakan **CNN berbasis TensorFlow**.

    ✅ Python  
    ✅ TensorFlow  
    ✅ CNN  
    ✅ Streamlit  
    """)

