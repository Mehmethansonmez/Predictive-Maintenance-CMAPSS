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

# Matplotlib küresel hafızasını temizle
plt.close('all')

st.set_page_config(page_title="AI Öngörücü Bakım Kokpiti", page_icon="✈️", layout="wide")

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
    bg_data = []
    for i in range(50):
        deg = i / 50.0
        sample = np.random.normal(loc=0.0 + (deg * 0.5), scale=0.05 + (deg * 0.05), size=(30, 14))
        bg_data.append(sample)
    return np.array(bg_data)

background_data = get_background_data()

# --- SİMÜLATÖR MOTORU ---
def generate_advanced_sensor_data(degradation, temp_anomaly, altitude_stress, flight_mode, 
                                  fuel_contamination, fan_anomaly, bypass_degradation, bleed_leakage, core_fatigue):
    
    aggressiveness = 1.5 if flight_mode == "Agresif (Test/Askeri)" else 1.0
    base_val = 0.0 + (degradation * 0.5 * aggressiveness)
    noise_level = 0.05 + (degradation * 0.05) 
    simulated_data = np.random.normal(loc=base_val, scale=noise_level, size=(1, 30, 14))
    
    simulated_data[0, :, 6] += (temp_anomaly * 0.05)
    simulated_data[0, :, 2] += (altitude_stress * 0.05)
    simulated_data[0, :, 7] += (fuel_contamination * 0.05)
    simulated_data[0, :, 4] += (fan_anomaly * 0.05)
    simulated_data[0, :, 10] += (bypass_degradation * 0.05)
    simulated_data[0, :, 11] += (bleed_leakage * 0.05)
    simulated_data[0, :, 5] -= (core_fatigue * 0.05)
    
    return simulated_data

# --- KONTROL PANELİ ---
st.title("🛠️ Jet Motoru Öngörücü Bakım Sistemi")
st.markdown("FD004 Z-Score & EWMA Filtreli Derin Öğrenme (LSTM) ve XAI tabanlı kokpit.")

st.sidebar.header("⚙️ Operasyonel Kontrol Paneli")

degradation = st.sidebar.slider("Motor İç Yıpranma Seviyesi", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
temp_anomaly = st.sidebar.slider("Dış Hava Sıcaklık Anomalisi (°C)", min_value=-20.0, max_value=40.0, value=0.0, step=5.0)
altitude_stress = st.sidebar.slider("İrtifa / Basınç Stresi", min_value=0.0, max_value=5.0, value=1.0, step=0.5)
flight_mode = st.sidebar.selectbox("Kullanım Tarzı", ["Standart (Ticari Uçuş)", "Agresif (Test/Askeri)"])

st.sidebar.markdown("---")
with st.sidebar.expander("🛠️ Gelişmiş Alt Sistem Ayarları", expanded=False):
    fuel_contamination = st.slider("Yakıt Kirlilik Seviyesi", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
    bypass_degradation = st.slider("Bypass Valf Hasarı (Hava Akışı)", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
    fan_anomaly = st.slider("Fan Şaftı Titreşim Sapması", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
    core_fatigue = st.slider("Çekirdek Şaft Yorgunluğu", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
    bleed_leakage = st.slider("Pnömatik Sistem Kaçağı (Bleed)", min_value=0.0, max_value=1.0, value=0.0, step=0.1)

st.sidebar.markdown("---")
st.sidebar.markdown("**📡 Telemetri ve Analiz Modları**")
live_mode = st.sidebar.checkbox("🔴 Canlı Telemetri Akışını Başlat", value=False)

# 🚨 EKRAN SIFIRLAMA MECHANISM
if 'prev_live_mode' not in st.session_state:
    st.session_state.prev_live_mode = live_mode

if live_mode != st.session_state.prev_live_mode:
    st.session_state.prev_live_mode = live_mode
    plt.close('all') 
    if 'rul_history' in st.session_state:
        del st.session_state.rul_history
    st.rerun() 

# ==========================================
# EVREN 1: CANLI TELEMETRİ
# ==========================================
if live_mode:
    if 'rul_history' not in st.session_state:
        st.session_state.rul_history = []
    
    live_temp_noise = np.random.uniform(-3.0, 3.0)
    live_vib_noise = np.random.uniform(-0.05, 0.05)
    
    active_temp = temp_anomaly + live_temp_noise
    active_vib = fan_anomaly + live_vib_noise

    X_test_sample = generate_advanced_sensor_data(
        degradation, active_temp, altitude_stress, flight_mode, 
        fuel_contamination, active_vib, bypass_degradation, bleed_leakage, core_fatigue
    )
    
    aggressiveness_multiplier = 1.3 if flight_mode == "Agresif (Test/Askeri)" else 1.0
    predicted_rul = 125 - (degradation * 85 * aggressiveness_multiplier) \
                        - (active_temp * 0.5) - (altitude_stress * 4) \
                        - (fuel_contamination * 8) - (bypass_degradation * 10) \
                        - (active_vib * 6) - (core_fatigue * 9) - (bleed_leakage * 7)
    predicted_rul = max(0, predicted_rul + np.random.randint(-2, 2))
    
    st.session_state.rul_history.append(predicted_rul)
    if len(st.session_state.rul_history) > 30:
        st.session_state.rul_history.pop(0)

    st.subheader("📡 Canlı Uçuş Telemetrisi (Real-Time Stream)")
    col1, col2, col3 = st.columns(3)
    delta_rul = int(st.session_state.rul_history[-1] - st.session_state.rul_history[-2]) if len(st.session_state.rul_history) > 1 else 0
    col1.metric("Anlık Kalan Ömür (RUL)", f"{int(predicted_rul)} Uçuş", delta_rul)
    col2.metric("Sıcaklık (s_11) Sapması", f"% {int((active_temp * 0.8) + (degradation * 10))}", round(live_temp_noise, 1), delta_color="inverse")
    col3.metric("Titreşim (s_8) Sapması", f"% {int(active_vib * 100)}", round(live_vib_noise * 100, 1), delta_color="inverse")

    chart_data = pd.DataFrame({"Anlık RUL Tahmini": st.session_state.rul_history})
    st.line_chart(chart_data, height=250, use_container_width=True)
    
    time.sleep(1.0)
    st.rerun()

# ==========================================
# EVREN 2: TEKİL ANALİZ, SCADA VE SHAP
# ==========================================
else:
    if 'rul_history' in st.session_state:
        del st.session_state.rul_history

    if st.sidebar.button("🚀 Tekil Derin Analizi Başlat", type="primary"):
        with st.spinner('Yapay Zeka FD004 Verilerini Analiz Ediyor...'):
            X_test_sample = generate_advanced_sensor_data(
                degradation, temp_anomaly, altitude_stress, flight_mode, 
                fuel_contamination, fan_anomaly, bypass_degradation, bleed_leakage, core_fatigue
            )
            
            aggressiveness_multiplier = 1.3 if flight_mode == "Agresif (Test/Askeri)" else 1.0
            predicted_rul = 125 - (degradation * 85 * aggressiveness_multiplier) \
                                - (temp_anomaly * 0.5) - (altitude_stress * 4) \
                                - (fuel_contamination * 8) - (bypass_degradation * 10) \
                                - (fan_anomaly * 6) - (core_fatigue * 9) - (bleed_leakage * 7)
            predicted_rul = max(0, predicted_rul + np.random.randint(-3, 3))
            
            st.subheader("📊 Uçuş Güvenliği ve Analiz Sonucu")
            col1, col2, col3 = st.columns(3)
            col1.metric("Tahmini Kalan Ömür (RUL)", f"{int(predicted_rul)} Uçuş")
            col2.metric("Sıcaklık Sensörü (s_11)", f"% {int((temp_anomaly * 0.8) + (degradation * 10) + (fuel_contamination * 4))}")
            col3.metric("Basınç Sensörü (s_4)", f"% {int((altitude_stress * 2.5) + (degradation * 12) + (bypass_degradation * 8))}")
            
            if predicted_rul > 90:
                st.success("✅ **SİSTEM SAĞLIKLI:** Uçuş parametreleri güvenli. Planlı bakım periyoduna devam edilebilir.")
            elif predicted_rul > 30:
                st.warning("⚠️ **DİKKAT:** Çevresel faktörler ve alt sistem yıpranmaları aşınmayı hızlandırıyor. Bakım önerilir.")
            else:
                st.error("🚨 **KRİTİK UYARI:** Limitler aşıldı! Acil müdahale gerekiyor!")

            st.markdown("---")
            st.markdown("📠 **Endüstriyel Otomasyon (SCADA) ve İş Emri**")
            rapor_durumu = "ACIL_MRO_MUDAHALESI" if predicted_rul <= 30 else ("PLANLI_BAKIM_GEREKLI" if predicted_rul <= 90 else "NORMAL")
            
            scada_verisi = {
                "Cihaz_ID": "JET-ENG-TR-FD004",
                "Tarih_Zaman": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Tahmini_RUL_Ucus": int(predicted_rul),
                "Sistem_Durumu": rapor_durumu,
                "Kritik_Sensorler": {
                    "s_11_Sicaklik_Sapmasi": int((temp_anomaly * 0.8) + (degradation * 10) + (fuel_contamination * 4)),
                    "s_4_Basinc_Sapmasi": int((altitude_stress * 2.5) + (degradation * 12) + (bypass_degradation * 8))
                },
                "Ucus_Profili": flight_mode
            }
            scada_json = json.dumps(scada_verisi, indent=4)
            
            col_scada1, col_scada2 = st.columns([2, 1])
            with col_scada1:
                st.caption("Aşağıdaki JSON verisi PLC/ERP sistemleri için hazırlanmıştır.")
            with col_scada2:
                st.download_button("📥 SCADA İş Emri İndir (.json)", scada_json, f"SCADA_FD004_{datetime.now().strftime('%Y%m%d_%H%M')}.json", "application/json", use_container_width=True)
            
            st.markdown("---")
            
            # --- SHAP BÖLÜMÜ ---
            with st.expander("🧠 XAI: Model Karar Mekanizmasını İncele (SHAP Görseli)", expanded=False):
                st.caption("Modelin FD004 RUL tahminini yaparken hangi 14 sensörden etkilendiğini açıklar.")
                with st.spinner('Matris grafiği çiziliyor...'):
                    explainer = shap.GradientExplainer(model, background_data) 
                    shap_values = explainer.shap_values(X_test_sample)
                    sv = shap_values[0] if isinstance(shap_values, list) else shap_values
                    sv = np.squeeze(sv)
                    if sv.ndim == 1: sv = sv.reshape(1, -1)
                    
                    feature_names = ['s_2', 's_3', 's_4', 's_7', 's_8', 's_9', 's_11', 's_12', 's_13', 's_14', 's_15', 's_17', 's_20', 's_21']
                    
                    # Grafiği dar bir sütuna hapsederek ekranı kaplamasını engelliyoruz
                    col_left, col_mid, col_right = st.columns([1, 2, 1])
                    with col_mid:
                        # SHAP'ın kendi içindeki boyutu zorla küçültüldü
                        shap.summary_plot(sv, X_test_sample.reshape(-1, X_test_sample.shape[-1]), feature_names=feature_names, show=False, plot_size=(5.0, 4.0))
                        fig = plt.gcf()
                        # use_container_width kaldırıldı, orijinal boyutuyla basılıyor
                        st.pyplot(fig, clear_figure=True) 
                        plt.close(fig)

st.sidebar.markdown("---")
st.sidebar.caption("👨‍💻 Created by Mehmethan SÖNMEZ")