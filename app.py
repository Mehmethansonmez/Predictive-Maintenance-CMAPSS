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

# Matplotlib global ayarları
plt.close('all')
st.set_page_config(page_title="Prediktif Bakım Ağı", page_icon="⚙️", layout="wide")

# --- 1. SİSTEM BAŞLATMA (Yapay Zeka ve Hafıza) ---
@st.cache_resource
def load_ai():
    return load_model('models/cmapss_fd004_lstm_v1.h5') if os.path.exists('models/cmapss_fd004_lstm_v1.h5') else load_model('/content/cmapss_fd004_lstm_v1.h5')

model = load_ai()

# Gerçek Yapay Zeka için 30 Zaman Adımlık Kayan Hafıza (Rolling Buffer)
if 'sensor_buffer' not in st.session_state:
    st.session_state.sensor_buffer = np.zeros((30, 14)) # Motor ilk çalıştığında tüm sapmalar sıfır
if 'rul_history' not in st.session_state:
    st.session_state.rul_history = []

# --- 2. SENSÖR GÜNCELLEME MOTORU ---
def update_sensor_buffer(degradation, temp, alt, fuel, vib, bypass, bleed, core):
    # Yeni bir anlık sensör okuması (14 parametre)
    new_reading = np.zeros(14)
    new_reading[6] = temp * 0.05 + np.random.normal(0, 0.02)
    new_reading[2] = alt * 0.05 + np.random.normal(0, 0.02)
    new_reading[7] = fuel * 0.05
    new_reading[4] = vib * 0.05 + np.random.normal(0, 0.01)
    new_reading[10] = bypass * 0.05
    new_reading[11] = bleed * 0.05
    new_reading[5] = -(core * 0.05)
    
    # Eski verileri 1 adım yukarı kaydır ve yeni okumayı en alta ekle
    st.session_state.sensor_buffer = np.roll(st.session_state.sensor_buffer, -1, axis=0)
    st.session_state.sensor_buffer[-1] = new_reading + (degradation * 0.5)

# --- 3. KONTROL PANELİ ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2103/2103130.png", width=50) # Şık bir ikon
st.sidebar.header("Uçuş Parametreleri")

deg_val = st.sidebar.slider("Yıpranma", 0.0, 1.0, 0.2)
temp_val = st.sidebar.slider("Sıcaklık Anomalisi", -20.0, 40.0, 0.0)
alt_val = st.sidebar.slider("Basınç Stresi", 0.0, 5.0, 1.0)
vib_val = st.sidebar.slider("Titreşim", 0.0, 1.0, 0.0)

st.sidebar.markdown("---")
is_live = st.sidebar.toggle("🔴 Canlı Telemetri Modu", value=False)

# --- 4. SEKME (TAB) MİMARİSİ (Arayüz çöküşlerini engelleyen yapı) ---
st.title("Jet Motoru Derin Öğrenme Kokpiti")
tab_telemetri, tab_xai, tab_scada = st.tabs(["📡 Canlı Akış", "🧠 XAI (SHAP) Analizi", "📠 SCADA / ERP"])

# === SEKME 1: CANLI TELEMETRİ ===
with tab_telemetri:
    st.markdown("### Sensör Zaman Serisi ve RUL Akışı")
    
    # Ekranda titremeden güncellenmesi için placeholder'lar
    metric_cols = st.columns(3)
    m1 = metric_cols[0].empty()
    m2 = metric_cols[1].empty()
    m3 = metric_cols[2].empty()
    chart_spot = st.empty()
    
    if is_live:
        # Arka planda kesintisiz çalışan akış döngüsü
        update_sensor_buffer(deg_val, temp_val, alt_val, 0.0, vib_val, 0.0, 0.0, 0.0)
        
        # Buffer'ı (30x14) LSTM'in istediği 3 Boyutlu Tensöre (1x30x14) çevirip GERÇEK tahmin alıyoruz!
        lstm_input = np.expand_dims(st.session_state.sensor_buffer, axis=0)
        predicted_rul = int(model.predict(lstm_input, verbose=0)[0][0])
        
        st.session_state.rul_history.append(predicted_rul)
        if len(st.session_state.rul_history) > 50: st.session_state.rul_history.pop(0)
        
        m1.metric("Gerçek Zamanlı RUL", f"{predicted_rul} Döngü")
        m2.metric("Egzoz Sıcaklık Trendi", f"% {int(temp_val + deg_val*10)}")
        m3.metric("Şaft Titreşimi", f"% {int(vib_val*100)}")
        
        chart_spot.line_chart(pd.DataFrame({"RUL": st.session_state.rul_history}), height=300)
        
        time.sleep(1.0)
        st.rerun() # Sadece bu sekmeyi ve verileri günceller
    else:
        st.info("Canlı yayın kapalı. RUL modelini incelemek için sol menüden şalteri açın.")

# === SEKME 2: XAI / SHAP (Sadece tıklandığında hesaplanır, kasmayı önler) ===
with tab_xai:
    st.markdown("### Karar Mekanizması Açıklayıcısı")
    if is_live:
        st.warning("⚠️ Lütfen derinlemesine SHAP analizi için telemetriyi durdurun.")
    else:
        if st.button("Güncel Sensör Durumu İçin SHAP Üret"):
            with st.spinner("LSTM ağırlıkları analiz ediliyor..."):
                lstm_input = np.expand_dims(st.session_state.sensor_buffer, axis=0)
                bg_data = np.random.normal(0, 0.05, (50, 30, 14)) # İzole background
                
                explainer = shap.GradientExplainer(model, bg_data)
                sv = explainer.shap_values(lstm_input)[0]
                if sv.ndim == 1: sv = sv.reshape(1, -1)
                
                features = ['s_2', 's_3', 's_4', 's_7', 's_8', 's_9', 's_11', 's_12', 's_13', 's_14', 's_15', 's_17', 's_20', 's_21']
                fig, ax = plt.subplots(figsize=(5, 3))
                shap.summary_plot(sv, lstm_input.reshape(-1, 14), feature_names=features, show=False)
                st.pyplot(fig, clear_figure=True)
                plt.close(fig)

# === SEKME 3: SCADA / ERP ===
with tab_scada:
    st.markdown("### Endüstriyel İş Emri Oluşturucu")
    current_rul = st.session_state.rul_history[-1] if st.session_state.rul_history else 125
    durum = "KRİTİK_BAKIM" if current_rul < 40 else "NORMAL"
    
    st.json({
        "Cihaz": "FD004-AERO",
        "Tarih": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Kalan_Omur_Tahmini": current_rul,
        "Sistem_Durumu": durum
    })
    st.button("Veritabanına Kaydet", disabled=(durum=="NORMAL"))

st.sidebar.markdown("---")
st.sidebar.caption("👨‍💻 Developed by Mehmethan SÖNMEZ | SUBU")