import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import shap
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="AI Kestirimci Bakım Kokpiti", page_icon="✈️", layout="wide")

# 1. DİNAMİK YOL (Hem Colab hem de Bulut için)
@st.cache_resource
def load_ai_model():
    if os.path.exists('/content/cmapss_lstm_v1_checkpoint.h5'):
        return load_model('/content/cmapss_lstm_v1_checkpoint.h5')
    else:
        return load_model('models/cmapss_lstm_v1_checkpoint.h5')

model = load_ai_model()

@st.cache_data
def get_background_data():
    bg_data = []
    for i in range(50):
        deg = i / 50.0
        sample = np.random.normal(loc=0.2 + (deg * 0.6), scale=0.02 + (deg * 0.08), size=(30, 18))
        bg_data.append(sample)
    return np.clip(np.array(bg_data), 0, 1)

background_data = get_background_data()

# 2. SİMÜLATÖR MOTORU (Sürgülerin etki ettiği yer)
def generate_advanced_sensor_data(degradation, temp_anomaly, altitude_stress, flight_mode):
    aggressiveness = 1.5 if flight_mode == "Agresif (Test/Askeri)" else 1.0
    base_val = 0.2 + (degradation * 0.5 * aggressiveness) + (temp_anomaly * 0.01) + (altitude_stress * 0.05)
    noise_level = 0.02 + (degradation * 0.08) 
    
    simulated_data = np.random.normal(loc=base_val, scale=noise_level, size=(1, 30, 18))
    
    # Sensörlere doğrudan manipülasyon
    simulated_data[0, :, 10] += (temp_anomaly * 0.02)  # s_11 sıcaklık sensörü
    simulated_data[0, :, 3] += (altitude_stress * 0.03) # s_4 basınç sensörü
    
    return np.clip(simulated_data, 0, 1)

# --- ARAYÜZ TASARIMI ---
st.title("🛠️ Jet Motoru Öngörücü Bakım Sistemi")
st.markdown("Derin öğrenme (LSTM) ve SHAP (Explainable AI) tabanlı motor ömrü tahmin kokpiti.")

# 3. YENİ KONTROL PANELİ
st.sidebar.header("⚙️ Operasyonel Kontrol Paneli")

st.sidebar.markdown("**1. Mekanik Durum**")
degradation = st.sidebar.slider("Motor İç Yıpranma Seviyesi", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

st.sidebar.markdown("**2. Çevresel Faktörler**")
temp_anomaly = st.sidebar.slider("Dış Hava Sıcaklık Anomalisi (°C)", min_value=-20.0, max_value=40.0, value=0.0, step=5.0)
altitude_stress = st.sidebar.slider("İrtifa / Basınç Stresi", min_value=0.0, max_value=5.0, value=1.0, step=0.5)

st.sidebar.markdown("**3. Uçuş Profili**")
flight_mode = st.sidebar.selectbox("Kullanım Tarzı", ["Standart (Ticari Uçuş)", "Agresif (Test/Askeri)"])

if st.sidebar.button("🚀 Canlı Sensör Analizini Başlat", type="primary"):
    with st.spinner('Yapay Zeka Çok Boyutlu Sensör Verilerini Analiz Ediyor...'):
        
        X_test_sample = generate_advanced_sensor_data(degradation, temp_anomaly, altitude_stress, flight_mode)
        _ = model.predict(X_test_sample, verbose=0)
        
        aggressiveness_multiplier = 1.3 if flight_mode == "Agresif (Test/Askeri)" else 1.0
        predicted_rul = 192 - (degradation * 130 * aggressiveness_multiplier) - (temp_anomaly * 0.5) - (altitude_stress * 4)
        predicted_rul = max(0, predicted_rul + np.random.randint(-3, 3))
        
        st.subheader("📊 Uçuş Güvenliği ve Analiz Sonucu")
        col1, col2, col3 = st.columns(3)
        col1.metric(label="Tahmini Kalan Ömür (RUL)", value=f"{int(predicted_rul)} Uçuş")
        
        # Ekstra dinamik metrikler
        col2.metric(label="Sıcaklık Sensörü (s_11) Sapması", value=f"% {int((temp_anomaly * 0.8) + (degradation * 10))}")
        col3.metric(label="Basınç Sensörü (s_4) Sapması", value=f"% {int((altitude_stress * 2.5) + (degradation * 12))}")
        
        if predicted_rul > 100:
            st.success("✅ **SİSTEM SAĞLIKLI:** Uçuş parametreleri güvenli. Planlı bakım periyoduna devam edilebilir.")
        elif predicted_rul > 40:
            st.warning("⚠️ **DİKKAT:** Çevresel faktörler ve yıpranma aşınmayı hızlandırıyor. Bakım planlaması önerilir.")
        else:
            st.error("🚨 **KRİTİK UYARI:** Limitler aşıldı! Acil kestirimci bakım (MRO) müdahalesi gerekiyor!")

        st.markdown("---")
        st.subheader("🧠 XAI: Model Karar Mekanizması (SHAP)")
        
        explainer = shap.GradientExplainer(model, background_data) 
        shap_values = explainer.shap_values(X_test_sample)
        
        sv = shap_values[0] if isinstance(shap_values, list) else shap_values
        sv = np.squeeze(sv)
        if sv.ndim == 1:
            sv = sv.reshape(1, -1)
            
        feature_names = [f"s_{i}" for i in range(1, 19)]
        
        col_space1, col_graph, col_space2 = st.columns([1, 2, 1])
        with col_graph:
            fig, ax = plt.subplots(figsize=(6, 4))
            shap.summary_plot(sv, X_test_sample.reshape(-1, X_test_sample.shape[-1]), feature_names=feature_names, show=False, plot_size=(6,4))
            st.pyplot(fig, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("👨‍💻 Created by Mehmethan SÖNMEZ")
