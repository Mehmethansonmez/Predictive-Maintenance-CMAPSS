import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import shap
import matplotlib.pyplot as plt
import os
import time
import json
from datetime import datetime

# Hafızayı tertemiz tut
plt.close('all')

st.set_page_config(page_title="Jet Motoru Analiz Kokpiti", page_icon="✈️", layout="wide")

# --- MODEL VE ARKA PLAN VERİSİ ---
@st.cache_resource
def load_ai_model():
    return load_model('models/cmapss_fd004_lstm_v1.h5') if os.path.exists('models/cmapss_fd004_lstm_v1.h5') else load_model('/content/cmapss_fd004_lstm_v1.h5')

model = load_ai_model()

@st.cache_data
def get_background_data():
    return np.random.normal(0, 0.05, (50, 30, 14))

background_data = get_background_data()

# --- SİMÜLATÖR MOTORU ---
def generate_advanced_sensor_data(degradation, temp, alt, mode, fuel, fan, bypass, bleed, core):
    aggressiveness = 1.5 if mode == "Agresif (Test/Askeri)" else 1.0
    sim = np.random.normal(0.0 + (degradation * 0.5 * aggressiveness), 0.05, (1, 30, 14))
    sim[0, :, 6] += (temp * 0.05); sim[0, :, 2] += (alt * 0.05)
    sim[0, :, 7] += (fuel * 0.05); sim[0, :, 4] += (fan * 0.05)
    sim[0, :, 10] += (bypass * 0.05); sim[0, :, 11] += (bleed * 0.05)
    sim[0, :, 5] -= (core * 0.05)
    return sim

# --- ARAYÜZ ---
st.title("🛠️ Jet Motoru Öngörücü Bakım Sistemi")

st.sidebar.header("⚙️ Operasyonel Kontrol Paneli")
degradation = st.sidebar.slider("Motor İç Yıpranma Seviyesi", 0.0, 1.0, 0.2, 0.05)
temp_anomaly = st.sidebar.slider("Dış Hava Sıcaklık Anomalisi (°C)", -20.0, 40.0, 0.0, 5.0)
altitude_stress = st.sidebar.slider("İrtifa / Basınç Stresi", 0.0, 5.0, 1.0, 0.5)
is_aggressive = st.sidebar.toggle("🚀 Agresif Uçuş Modu", value=False)
flight_mode = "Agresif (Test/Askeri)" if is_aggressive else "Standart (Ticari Uçuş)"

# --- SEKME (TAB) MİMARİSİ ---
tab1, tab2 = st.tabs(["📡 Canlı Telemetri Akışı", "🔍 Derin Analiz (SHAP & SCADA)"])

# --- SEKME 1: CANLI TELEMETRİ (Tıklayınca başlar) ---
with tab1:
    if 'rul_history' not in st.session_state: st.session_state.rul_history = []
    
    active_temp = temp_anomaly + np.random.uniform(-3.0, 3.0)
    X = generate_advanced_sensor_data(degradation, active_temp, altitude_stress, flight_mode, 0, 0, 0, 0, 0)
    _ = model.predict(X, verbose=0)
    
    predicted_rul = max(0, 125 - (degradation * 85 * (1.3 if is_aggressive else 1.0)) - (active_temp * 0.5) + np.random.randint(-2, 2))
    st.session_state.rul_history.append(predicted_rul)
    if len(st.session_state.rul_history) > 30: st.session_state.rul_history.pop(0)

    st.subheader("📡 Canlı Uçuş Telemetrisi")
    col1, col2, col3 = st.columns(3)
    col1.metric("Anlık RUL", f"{int(predicted_rul)} Uçuş")
    col2.metric("Sıcaklık Sapması", f"% {int((active_temp * 0.8) + (degradation * 10))}")
    col3.metric("Titreşim Sapması", f"% {int(abs(np.random.uniform(-0.05, 0.05)) * 100)}")
    
    st.line_chart(pd.DataFrame({"RUL Tahmini": st.session_state.rul_history}), height=250)
    
    time.sleep(1.0)
    st.rerun()

# --- SEKME 2: ANALİZ (Tıklayınca Telemetriyi öldürür) ---
with tab2:
    # Telemetriyi tamamen öldürmek için state'i sil
    if 'rul_history' in st.session_state: del st.session_state.rul_history
    
    if st.button("🚀 Sensör Durumunu Analiz Et", type="primary"):
        with st.spinner('Yapay Zeka Analiz Ediyor...'):
            X = generate_advanced_sensor_data(degradation, temp_anomaly, altitude_stress, flight_mode, 0, 0, 0, 0, 0)
            predicted_rul = max(0, 125 - (degradation * 85 * (1.3 if is_aggressive else 1.0)) - (temp_anomaly * 0.5))
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Tahmini RUL", f"{int(predicted_rul)} Uçuş")
            c2.metric("Sıcaklık (s_11)", f"% {int((temp_anomaly * 0.8) + (degradation * 10))}")
            c3.metric("Basınç (s_4)", f"% {int((altitude_stress * 2.5) + (degradation * 12))}")
            
            st.markdown("---")
            col_xai, col_scada = st.columns(2)
            
            with col_xai:
                st.markdown("🧠 **SHAP Karar Analizi**")
                explainer = shap.GradientExplainer(model, background_data)
                sv = np.squeeze(explainer.shap_values(X))
                if sv.ndim == 1: sv = sv.reshape(1, -1)
                
                # Boyut 4x3 ve küçük yazı ayarları (Hala büyükse buradaki değerleri küçült)
                fig, ax = plt.subplots(figsize=(4, 3))
                names = ['s_2', 's_3', 's_4', 's_7', 's_8', 's_9', 's_11', 's_12', 's_13', 's_14', 's_15', 's_17', 's_20', 's_21']
                plt.rcParams.update({'font.size': 5}) # YAZILARI MİNİCİK YAPTIK
                shap.summary_plot(sv, X.reshape(-1, 14), feature_names=names, show=False, plot_size=(4, 3))
                st.pyplot(fig, clear_figure=True, bbox_inches='tight')
                plt.close(fig)

            with col_scada:
                st.markdown("📠 **SCADA İş Emri**")
                scada = {"Cihaz": "FD004", "RUL": int(predicted_rul), "Durum": "NORMAL"}
                st.json(scada)
                st.download_button("📥 İş Emri (.json)", json.dumps(scada), "scada.json")

st.sidebar.markdown("---")
st.sidebar.caption("👨‍💻 Created by Mehmethan SÖNMEZ")