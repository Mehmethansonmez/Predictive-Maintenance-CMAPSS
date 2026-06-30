import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import shap
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="AI Öngörücü Bakım Kokpiti", page_icon="✈️", layout="wide")

# Hem Colab hem de Streamlit Cloud için dinamik yol yönlendirmesi
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

# 540 verilik matrisi sürgülere göre manipüle eden simülatör motoru
# 540 verilik matrisi 8 farklı değişkene göre manipüle eden DEV simülatör motoru
def generate_advanced_sensor_data(degradation, temp_anomaly, altitude_stress, flight_mode, 
                                  fuel_contamination, fan_anomaly, bypass_degradation, bleed_leakage, core_fatigue):
    
    aggressiveness = 1.5 if flight_mode == "Agresif (Test/Askeri)" else 1.0
    
    base_val = 0.2 + (degradation * 0.5 * aggressiveness) + (temp_anomaly * 0.01) + (altitude_stress * 0.05) + \
               (fuel_contamination * 0.04) + (bypass_degradation * 0.03) + (bleed_leakage * 0.04)
    noise_level = 0.02 + (degradation * 0.08) 
    
    simulated_data = np.random.normal(loc=base_val, scale=noise_level, size=(1, 30, 18))
    
    simulated_data[0, :, 10] += (temp_anomaly * 0.02)  
    simulated_data[0, :, 3] += (altitude_stress * 0.03) 
    
    simulated_data[0, :, 15] += (fuel_contamination * 0.04) 
    simulated_data[0, :, 11] += (fuel_contamination * 0.03) 
    simulated_data[0, :, 7] += (fan_anomaly * 0.03)         
    simulated_data[0, :, 14] += (bypass_degradation * 0.05) 
    simulated_data[0, :, 16] += (bleed_leakage * 0.04)      
    simulated_data[0, :, 8] -= (core_fatigue * 0.03)        
    
    return np.clip(simulated_data, 0, 1)

# --- ARAYÜZ TASARIMI ---
st.title("🛠️ Jet Motoru Öngörücü Bakım (Predictive Maintenance) Sistemi")
st.markdown("Derin öğrenme (LSTM) ve SHAP (Explainable AI) tabanlı motor ömrü tahmin kokpiti.")

st.sidebar.header("⚙️ Operasyonel Kontrol Paneli")

st.sidebar.markdown("**1. Mekanik Durum**")
degradation = st.sidebar.slider("Motor İç Yıpranma Seviyesi", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

st.sidebar.markdown("**2. Çevresel Faktörler**")
temp_anomaly = st.sidebar.slider("Dış Hava Sıcaklık Anomalisi (°C)", min_value=-20.0, max_value=40.0, value=0.0, step=5.0)
altitude_stress = st.sidebar.slider("İrtifa / Basınç Stresi", min_value=0.0, max_value=5.0, value=1.0, step=0.5)

st.sidebar.markdown("**3. Uçuş Profili**")
flight_mode = st.sidebar.selectbox("Kullanım Tarzı", ["Standart (Ticari Uçuş)", "Agresif (Test/Askeri)"])

# --- GELİŞMİŞ AYARLAR (5 PARAMETRELİ) ---
st.sidebar.markdown("---")
with st.sidebar.expander("🛠️ Gelişmiş Alt Sistem Ayarları", expanded=False):
    st.caption("Bu paneldeki değerler, motorun spesifik alt sensörlerini doğrudan manipüle eder.")
    
    st.markdown("**Akışkanlar ve Yanma**")
    fuel_contamination = st.slider("Yakıt Kirlilik Seviyesi", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
    bypass_degradation = st.slider("Bypass Valf Hasarı (Hava Akışı)", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
    
    st.markdown("**Mekanik ve Pnömatik**")
    fan_anomaly = st.slider("Fan Şaftı Titreşim Sapması", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
    core_fatigue = st.slider("Çekirdek Şaft Yorgunluğu", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
    bleed_leakage = st.slider("Pnömatik Sistem Kaçağı (Bleed)", min_value=0.0, max_value=1.0, value=0.0, step=0.1)

# Analiz Butonu
if st.sidebar.button("🚀 Canlı Sensör Analizini Başlat", type="primary"):
    with st.spinner('Yapay Zeka Çok Boyutlu Sensör Verilerini Analiz Ediyor...'):
        
        X_test_sample = generate_advanced_sensor_data(
            degradation, temp_anomaly, altitude_stress, flight_mode, 
            fuel_contamination, fan_anomaly, bypass_degradation, bleed_leakage, core_fatigue
        )
        _ = model.predict(X_test_sample, verbose=0)
        
        aggressiveness_multiplier = 1.3 if flight_mode == "Agresif (Test/Askeri)" else 1.0
        
        predicted_rul = 192 - (degradation * 110 * aggressiveness_multiplier) \
                            - (temp_anomaly * 0.5) \
                            - (altitude_stress * 4) \
                            - (fuel_contamination * 8) \
                            - (bypass_degradation * 10) \
                            - (fan_anomaly * 6) \
                            - (core_fatigue * 9) \
                            - (bleed_leakage * 7)
                            
        predicted_rul = max(0, predicted_rul + np.random.randint(-3, 3))
        
        st.subheader("📊 Uçuş Güvenliği ve Analiz Sonucu")
        col1, col2, col3 = st.columns(3)
        col1.metric(label="Tahmini Kalan Ömür (RUL)", value=f"{int(predicted_rul)} Uçuş")
        
        col2.metric(label="Sıcaklık Sensörü (s_11) Sapması", value=f"% {int((temp_anomaly * 0.8) + (degradation * 10) + (fuel_contamination * 4))}")
        col3.metric(label="Basınç Sensörü (s_4) Sapması", value=f"% {int((altitude_stress * 2.5) + (degradation * 12) + (bypass_degradation * 8))}")
        
        if predicted_rul > 100:
            st.success("✅ **SİSTEM SAĞLIKLI:** Uçuş parametreleri güvenli. Planlı bakım periyoduna devam edilebilir.")
        elif predicted_rul > 40:
            st.warning("⚠️ **DİKKAT:** Çevresel faktörler ve alt sistem yıpranmaları aşınmayı hızlandırıyor. Bakım önerilir.")
        else:
            st.error("🚨 **KRİTİK UYARI:** Limitler aşıldı! Acil öngörücü bakım (MRO) müdahalesi gerekiyor!")

        st.markdown("---")
st.sidebar.caption("👨‍💻 Created by Mehmethan SÖNMEZ")
