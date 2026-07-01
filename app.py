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

# Matplotlib hafızasını temizle
plt.close('all')

st.set_page_config(page_title="Jet Motoru Analiz Kokpiti", page_icon="✈️", layout="wide")

# --- MODEL VE ARKA PLAN VERİSİ ---
@st.cache_resource
def load_ai_model():
    if os.path.exists('/content/cmapss_fd004_lstm_v1.h5'):
        return load_model('/content/cmapss_fd004_lstm_v1.h5')
    else:
        return load_model('models/cmapss_fd004_lstm_v1.h5')

model = load_ai_model()

@st.cache_data
def get_background_data():
    return np.random.normal(0, 0.05, (50, 30, 14))

background_data = get_background_data()

# --- SİMÜLATÖR MOTORU ---
def generate_advanced_sensor_data(degradation, temp_anomaly, altitude_stress, flight_mode, 
                                  fuel_contamination, fan_anomaly, bypass_degradation, bleed_leakage, core_fatigue):
    aggressiveness = 1.5 if flight_mode == "Agresif (Test/Askeri)" else 1.0
    sim = np.random.normal(0.0 + (degradation * 0.5 * aggressiveness), 0.05, (1, 30, 14))
    sim[0, :, 6] += (temp_anomaly * 0.05); sim[0, :, 2] += (altitude_stress * 0.05)
    sim[0, :, 7] += (fuel_contamination * 0.05); sim[0, :, 4] += (fan_anomaly * 0.05)
    sim[0, :, 10] += (bypass_degradation * 0.05); sim[0, :, 11] += (bleed_leakage * 0.05)
    sim[0, :, 5] -= (core_fatigue * 0.05)
    return sim

# --- ARAYÜZ ---
st.title("🛠️ Jet Motoru Öngörücü Bakım Sistemi")

# Mod Seçimi (Switch yerine radyo ile daha stabil geçiş)
view_mode = st.radio("Mod Seçimi", ["📡 Canlı Telemetri Akışı", "🔍 Derin Analiz (SHAP & SCADA)"], 
                     horizontal=True, label_visibility="collapsed")

st.sidebar.header("⚙️ Operasyonel Kontrol Paneli")
degradation = st.sidebar.slider("Motor İç Yıpranma Seviyesi", 0.0, 1.0, 0.2, 0.05)
temp_anomaly = st.sidebar.slider("Dış Hava Sıcaklık Anomalisi (°C)", -20.0, 40.0, 0.0, 5.0)
altitude_stress = st.sidebar.slider("İrtifa / Basınç Stresi", 0.0, 5.0, 1.0, 0.5)
is_aggressive = st.sidebar.toggle("🚀 Agresif Uçuş Modu", value=False)
flight_mode = "Agresif (Test/Askeri)" if is_aggressive else "Standart (Ticari Uçuş)"

# --- CANLI TELEMETRİ ---
if view_mode == "📡 Canlı Telemetri Akışı":
    if 'rul_history' not in st.session_state: st.session_state.rul_history = []
    
    active_temp = temp_anomaly + np.random.uniform(-3.0, 3.0)
    active_vib = 0.0 + np.random.uniform(-0.05, 0.05) # Fan simülasyonu
    X = generate_advanced_sensor_data(degradation, active_temp, altitude_stress, flight_mode, 0, active_vib, 0, 0, 0)
    _ = model.predict(X, verbose=0)
    
    predicted_rul = max(0, 125 - (degradation * 85 * (1.3 if is_aggressive else 1.0)) - (active_temp * 0.5) + np.random.randint(-2, 2))
    st.session_state.rul_history.append(predicted_rul)
    if len(st.session_state.rul_history) > 30: st.session_state.rul_history.pop(0)

    st.subheader("📡 Canlı Uçuş Telemetrisi")
    col1, col2, col3 = st.columns(3)
    col1.metric("Anlık Kalan Ömür (RUL)", f"{int(predicted_rul)} Uçuş")
    col2.metric("Sıcaklık (s_11) Sapması", f"% {int((active_temp * 0.8) + (degradation * 10))}")
    col3.metric("Titreşim (s_8) Sapması", f"% {int(abs(active_vib) * 100)}")
    
    st.line_chart(pd.DataFrame({"Anlık RUL Tahmini": st.session_state.rul_history}), height=250)
    
    time.sleep(1.0)
    st.rerun()

# --- DERİN ANALİZ (SHAP + SCADA) ---
else:
    if 'rul_history' in st.session_state: del st.session_state.rul_history
    
    if st.button("🚀 Sensör Durumunu Analiz Et", type="primary"):
        with st.spinner('Yapay Zeka Analiz Ediyor...'):
            X = generate_advanced_sensor_data(degradation, temp_anomaly, altitude_stress, flight_mode, 0, 0, 0, 0, 0)
            predicted_rul = max(0, 125 - (degradation * 85 * (1.3 if is_aggressive else 1.0)) - (temp_anomaly * 0.5))
            
            st.subheader("📊 Uçuş Güvenliği Analiz Sonucu")
            c1, c2, c3 = st.columns(3)
            c1.metric("Tahmini Kalan Ömür (RUL)", f"{int(predicted_rul)} Uçuş")
            c2.metric("Sıcaklık (s_11) Sapması", f"% {int((temp_anomaly * 0.8) + (degradation * 10))}")
            c3.metric("Basınç (s_4) Sapması", f"% {int((altitude_stress * 2.5) + (degradation * 12))}")
            
            st.markdown("---")
            col_xai, col_scada = st.columns(2)
            
            with col_xai:
                st.markdown("🧠 **Yapay Zeka Karar Açıklayıcı (SHAP)**")
                explainer = shap.GradientExplainer(model, background_data)
                sv = np.squeeze(explainer.shap_values(X))
                if sv.ndim == 1: sv = sv.reshape(1, -1)
                
                # Grafik boyutunu zorla sabitle ve fontları küçült
                fig, ax = plt.subplots()
                fig.set_size_inches(4, 3) 
                names = ['s_2', 's_3', 's_4', 's_7', 's_8', 's_9', 's_11', 's_12', 's_13', 's_14', 's_15', 's_17', 's_20', 's_21']
                shap.summary_plot(sv, X.reshape(-1, 14), feature_names=names, show=False, plot_size=(4, 3))
                
                # Font boyutunu zorla
                for text in ax.get_yticklabels(): text.set_fontsize(6)
                for text in ax.get_xticklabels(): text.set_fontsize(6)
                
                st.pyplot(fig, clear_figure=True, bbox_inches='tight')
                plt.close(fig)

            with col_scada:
                st.markdown("📠 **Otomasyon ve İş Emri (SCADA)**")
                scada = {"Cihaz_ID": "JET-ENG-TR-FD004", "Tahmini_RUL": int(predicted_rul), "Durum": "NORMAL" if predicted_rul > 30 else "KRİTİK"}
                st.json(scada)
                st.download_button("📥 SCADA İş Emri İndir (.json)", json.dumps(scada), f"SCADA_{datetime.now().strftime('%Y%m%d')}.json")

st.sidebar.markdown("---")
st.sidebar.caption("👨‍💻 Created by Mehmethan SÖNMEZ")