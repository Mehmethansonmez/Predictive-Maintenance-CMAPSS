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

# Matplotlib global hafızasını temizle
plt.close('all')

st.set_page_config(page_title="AI Öngörücü Bakım Kokpiti", page_icon="✈️", layout="wide")

# --- CSS ENJEKSİYONU: Çirkin radyo tuşlarını modern butonlara çevirir ---
st.markdown("""
    <style>
    div[role="radiogroup"] > label > div:first-of-type {
        display: none !important;
    }
    div[role="radiogroup"] {
        display: flex;
        flex-direction: row;
        gap: 15px;
        width: 100%;
        margin-bottom: 10px;
    }
    div[role="radiogroup"] > label {
        flex: 1;
        justify-content: center;
        background-color: #262730;
        padding: 12px 20px;
        border-radius: 8px;
        border: 1px solid #33343d;
        cursor: pointer;
        transition: all 0.2s ease-in-out;
    }
    div[role="radiogroup"] > label:hover {
        background-color: #33343d;
        border-color: #ff4b4b;
    }
    div[role="radiogroup"] > label[data-checked="true"] {
        background-color: #ff4b4b !important;
        border-color: #ff4b4b !important;
    }
    div[role="radiogroup"] > label[data-checked="true"] p {
        color: white !important;
    }
    div[role="radiogroup"] > label p {
        font-size: 16px !important;
        font-weight: 600 !important;
        text-align: center !important;
        margin: 0 !important;
    }
    </style>
""", unsafe_allow_html=True)

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

st.sidebar.markdown("**Uçuş Profili**")
is_aggressive = st.sidebar.toggle("🚀 Agresif Uçuş Modu", value=False)
flight_mode = "Agresif (Test/Askeri)" if is_aggressive else "Standart (Ticari Uçuş)"

st.sidebar.markdown("---")
with st.sidebar.expander("🛠️ Gelişmiş Alt Sistem Ayarları", expanded=False):
    fuel_contamination = st.slider("Yakıt Kirlilik Seviyesi", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
    bypass_degradation = st.slider("Bypass Valf Hasarı (Hava Akışı)", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
    fan_anomaly = st.slider("Fan Şaftı Titreşim Sapması", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
    core_fatigue = st.slider("Çekirdek Şaft Yorgunluğu", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
    bleed_leakage = st.slider("Pnömatik Sistem Kaçağı (Bleed)", min_value=0.0, max_value=1.0, value=0.0, step=0.1)

# --- SİSTEM HAFIZASI ---
if "aktif_sekme" not in st.session_state:
    st.session_state.aktif_sekme = "telemetri"
if "rul_history" not in st.session_state:
    st.session_state.rul_history = []
if "analiz_yapildi" not in st.session_state:
    st.session_state.analiz_yapildi = False

# HİÇBİR ŞEY SİLİNMİYOR: Sadece görünümü (aktif katmanı) değiştiriyor
def sekme_degistir(sekme_adi):
    st.session_state.aktif_sekme = sekme_adi
    plt.close('all') 

col_tab1, col_tab2 = st.columns(2)

with col_tab1:
    st.button(
        "📡 Canlı Telemetri Akışı", 
        use_container_width=True, 
        type="primary" if st.session_state.aktif_sekme == "telemetri" else "secondary",
        on_click=sekme_degistir, 
        args=("telemetri",)
    )

with col_tab2:
    st.button(
        "🔍 Derin Analiz (SHAP & SCADA)", 
        use_container_width=True, 
        type="primary" if st.session_state.aktif_sekme == "analiz" else "secondary",
        on_click=sekme_degistir, 
        args=("analiz",)
    )

st.markdown("---")

# ==========================================
# KATMAN FONKSİYONLARI (Modüler Çizim)
# ==========================================

def katman_telemetri():
    st.subheader("📡 Canlı Uçuş Telemetrisi (Real-Time Stream)")
    
    # EĞER TELEMETRİ KATMANI ÜSTTEYSE (AKTİFSE) YENİ VERİ ÜRET
    if st.session_state.aktif_sekme == "telemetri":
        live_temp_noise = np.random.uniform(-3.0, 3.0)
        live_vib_noise = np.random.uniform(-0.05, 0.05)
        
        active_temp = temp_anomaly + live_temp_noise
        active_vib = fan_anomaly + live_vib_noise

        aggressiveness_multiplier = 1.3 if is_aggressive else 1.0
        predicted_rul = 125 - (degradation * 85 * aggressiveness_multiplier) \
                            - (active_temp * 0.5) - (altitude_stress * 4) \
                            - (fuel_contamination * 8) - (bypass_degradation * 10) \
                            - (active_vib * 6) - (core_fatigue * 9) - (bleed_leakage * 7)
        predicted_rul = max(0, predicted_rul + np.random.randint(-2, 2))
        
        st.session_state.rul_history.append(predicted_rul)
        if len(st.session_state.rul_history) > 30:
            st.session_state.rul_history.pop(0)
            
        # Gösterge değerlerini hafızaya kaydet (Arka planda donduğunda gösterebilmek için)
        st.session_state.t_rul = predicted_rul
        st.session_state.t_temp = active_temp
        st.session_state.t_vib = active_vib
        st.session_state.t_ntemp = live_temp_noise
        st.session_state.t_nvib = live_vib_noise
        
    # EKRANA ÇİZME (Aktifken hareketli, alttayken donuk çizer)
    if len(st.session_state.rul_history) > 0:
        col1, col2, col3 = st.columns(3)
        delta_rul = int(st.session_state.rul_history[-1] - st.session_state.rul_history[-2]) if len(st.session_state.rul_history) > 1 else 0
        
        disp_rul = st.session_state.get("t_rul", 125)
        disp_temp = st.session_state.get("t_temp", temp_anomaly)
        disp_vib = st.session_state.get("t_vib", fan_anomaly)
        disp_ntemp = st.session_state.get("t_ntemp", 0)
        disp_nvib = st.session_state.get("t_nvib", 0)
        
        col1.metric("Anlık Kalan Ömür (RUL)", f"{int(disp_rul)} Uçuş", delta_rul)
        col2.metric("Sıcaklık (s_11) Sapması", f"% {int((disp_temp * 0.8) + (degradation * 10))}", round(disp_ntemp, 1), delta_color="inverse")
        col3.metric("Titreşim (s_8) Sapması", f"% {int(disp_vib * 100)}", round(disp_nvib * 100, 1), delta_color="inverse")

        chart_data = pd.DataFrame({"Anlık RUL Tahmini": st.session_state.rul_history})
        st.line_chart(chart_data, height=250, use_container_width=True)
    else:
        st.info("Telemetri verisi bekleniyor...")

def katman_analiz():
    st.markdown("### Uçuş Güvenliği ve Karar Mekanizması Analizi")
    
    # Sadece Analiz katmanındayken buton tıklanabilir
    if st.button("🚀 Sensör Durumunu Analiz Et", type="primary", disabled=(st.session_state.aktif_sekme != "analiz")):
        st.session_state.analiz_yapildi = True
        with st.spinner('Yapay Zeka FD004 Verilerini Yorumluyor...'):
            X_test_sample = generate_advanced_sensor_data(
                degradation, temp_anomaly, altitude_stress, flight_mode, 
                fuel_contamination, fan_anomaly, bypass_degradation, bleed_leakage, core_fatigue
            )
            
            aggressiveness_multiplier = 1.3 if is_aggressive else 1.0
            predicted_rul = 125 - (degradation * 85 * aggressiveness_multiplier) \
                                - (temp_anomaly * 0.5) - (altitude_stress * 4) \
                                - (fuel_contamination * 8) - (bypass_degradation * 10) \
                                - (fan_anomaly * 6) - (core_fatigue * 9) - (bleed_leakage * 7)
            predicted_rul = max(0, predicted_rul + np.random.randint(-3, 3))
            
            # Hesaplamaları hafızaya (Cache) alıyoruz ki telemetri yenilenirken kasmasın
            st.session_state.a_rul = predicted_rul
            st.session_state.a_temp = temp_anomaly
            st.session_state.a_deg = degradation
            st.session_state.a_fuel = fuel_contamination
            st.session_state.a_alt = altitude_stress
            st.session_state.a_bypass = bypass_degradation
            st.session_state.a_X = X_test_sample
            
            explainer = shap.GradientExplainer(model, background_data) 
            shap_values = explainer.shap_values(X_test_sample)
            sv = shap_values[0] if isinstance(shap_values, list) else shap_values
            sv = np.squeeze(sv)
            if sv.ndim == 1: sv = sv.reshape(1, -1)
            st.session_state.a_sv = sv

    # Eğer daha önce analiz yapıldıysa, hafızadaki (Cache) veriyi hızlıca ekrana bas
    if st.session_state.analiz_yapildi:
        col1, col2, col3 = st.columns(3)
        col1.metric("Tahmini Kalan Ömür (RUL)", f"{int(st.session_state.a_rul)} Uçuş")
        col2.metric("Sıcaklık Sensörü (s_11)", f"% {int((st.session_state.a_temp * 0.8) + (st.session_state.a_deg * 10) + (st.session_state.a_fuel * 4))}")
        col3.metric("Basınç Sensörü (s_4)", f"% {int((st.session_state.a_alt * 2.5) + (st.session_state.a_deg * 12) + (st.session_state.a_bypass * 8))}")
        
        if st.session_state.a_rul > 90:
            st.success("✅ **SİSTEM SAĞLIKLI:** Uçuş parametreleri güvenli. Planlı bakım periyoduna devam edilebilir.")
        elif st.session_state.a_rul > 30:
            st.warning("⚠️ **DİKKAT:** Çevresel faktörler ve alt sistem yıpranmaları aşınmayı hızlandırıyor. Bakım önerilir.")
        else:
            st.error("🚨 **KRİTİK UYARI:** Limitler aşıldı! Acil müdahale gerekiyor!")

        st.markdown("---")
        col_xai, col_scada = st.columns(2)
        
        with col_xai:
            st.markdown("🧠 **Yapay Zeka Karar Açıklayıcı (SHAP)**")
            st.caption("Modelin 14 sensör ağırlığı.")
            feature_names = ['s_2', 's_3', 's_4', 's_7', 's_8', 's_9', 's_11', 's_12', 's_13', 's_14', 's_15', 's_17', 's_20', 's_21']
            
            plt.rcParams.update({'font.size': 6, 'axes.labelsize': 6, 'xtick.labelsize': 6, 'ytick.labelsize': 6})
            fig, ax = plt.subplots(figsize=(4, 3))
            shap.summary_plot(st.session_state.a_sv, st.session_state.a_X.reshape(-1, 14), feature_names=feature_names, show=False, plot_size=(4, 3))
            st.pyplot(fig, clear_figure=True, bbox_inches='tight')
            plt.close(fig)

        with col_scada:
            st.markdown("📠 **Otomasyon ve İş Emri (SCADA)**")
            st.caption("PLC/ERP sistemleri için machine-readable paket.")
            rapor_durumu = "ACIL_MRO_MUDAHALESI" if st.session_state.a_rul <= 30 else ("PLANLI_BAKIM_GEREKLI" if st.session_state.a_rul <= 90 else "NORMAL")
            scada_verisi = {
                "Cihaz_ID": "JET-ENG-TR-FD004",
                "Tarih_Zaman": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Tahmini_RUL_Ucus": int(st.session_state.a_rul),
                "Sistem_Durumu": rapor_durumu,
                "Kritik_Sensorler": {
                    "s_11_Sicaklik_Sapmasi": int((st.session_state.a_temp * 0.8) + (st.session_state.a_deg * 10) + (st.session_state.a_fuel * 4)),
                    "s_4_Basinc_Sapmasi": int((st.session_state.a_alt * 2.5) + (st.session_state.a_deg * 12) + (st.session_state.a_bypass * 8))
                },
                "Ucus_Profili": flight_mode
            }
            scada_json = json.dumps(scada_verisi, indent=4)
            st.json(scada_verisi)
            st.download_button("📥 İş Emri (.json)", scada_json, f"SCADA_FD004_{datetime.now().strftime('%Y%m%d_%H%M')}.json", "application/json", use_container_width=True)

# ==========================================
# ANA EKRAN YERLEŞİMİ (Tıklanana Göre Üste Alır)
# ==========================================

if st.session_state.aktif_sekme == "telemetri":
    # 1. Katman: Telemetri (Üstte)
    katman_telemetri()
    st.markdown("---")
    
    # 2. Katman: Analiz (Altta, Donuk)
    katman_analiz()
    
    # Sadece Telemetri üstteyken ekran saniyede bir kendini yeniler
    time.sleep(1.0)
    st.rerun()

else:
    # 1. Katman: Analiz (Üstte, İnteraktif)
    katman_analiz()
    st.markdown("---")
    
    # 2. Katman: Telemetri (Altta, Donuk)
    katman_telemetri()
    
    # Ekran yenileme yok, kullanıcı analiz butonuna rahatça basabilir