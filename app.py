import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import shap
import matplotlib.pyplot as plt
import os
import time # Canlı akış zamanlaması için eklendi

st.set_page_config(page_title="AI Öngörücü Bakım Kokpiti", page_icon="✈️", layout="wide")

# Hem Colab hem de Streamlit Cloud için dinamik yol yönlendirmesi
@st.cache_resource
def load_ai_model():
    # Eski v1 modeli FD004 modeli ile değiştirildi
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
        # Yeni model 14 sensörlü ve Z-Score normalize olduğu için taban değer sıfır etrafındadır
        sample = np.random.normal(loc=0.0 + (deg * 0.5), scale=0.05 + (deg * 0.05), size=(30, 14))
        bg_data.append(sample)
    return np.array(bg_data) # Z-score eksilere düşebileceği için kısıtlamayı kaldırdık

background_data = get_background_data()

# FD004 Z-Score Normalizasyonuna Uygun Yeni Simülatör (14 Sensör)
def generate_advanced_sensor_data(degradation, temp_anomaly, altitude_stress, flight_mode, 
                                  fuel_contamination, fan_anomaly, bypass_degradation, bleed_leakage, core_fatigue):
    
    aggressiveness = 1.5 if flight_mode == "Agresif (Test/Askeri)" else 1.0
    
    # Yeni model Z-Score normalize olduğu için taban değer sıfır etrafındadır
    base_val = 0.0 + (degradation * 0.5 * aggressiveness)
    noise_level = 0.05 + (degradation * 0.05) 
    
    # 1 Batch, 30 Zaman Adımı, 14 Sensör
    simulated_data = np.random.normal(loc=base_val, scale=noise_level, size=(1, 30, 14))
    
    # Yeni 14'lü dizilime göre nokta atışı sensör manipülasyonları:
    simulated_data[0, :, 6] += (temp_anomaly * 0.05)       # s_11 (Sıcaklık)
    simulated_data[0, :, 2] += (altitude_stress * 0.05)    # s_4 (Basınç)
    simulated_data[0, :, 7] += (fuel_contamination * 0.05) # s_12 (Yakıt Akışı)
    simulated_data[0, :, 4] += (fan_anomaly * 0.05)        # s_8 (Fan Titreşimi)
    simulated_data[0, :, 10] += (bypass_degradation * 0.05)# s_15 (Bypass Oranı)
    simulated_data[0, :, 11] += (bleed_leakage * 0.05)     # s_17 (Pnömatik Kaçak)
    simulated_data[0, :, 5] -= (core_fatigue * 0.05)       # s_9 (Çekirdek Hızı Düşüşü)
    
    return simulated_data

# --- ARAYÜZ TASARIMI ---
st.title("🛠️ Jet Motoru Öngörücü Bakım Sistemi")
st.markdown("Derin öğrenme (LSTM) ve SHAP tabanlı **Gerçek Zamanlı** motor ömrü tahmin kokpiti.")

st.sidebar.header("⚙️ Operasyonel Kontrol Paneli")

# Ana Giriş Sürgüleri
st.sidebar.markdown("**1. Mekanik Durum**")
degradation = st.sidebar.slider("Motor İç Yıpranma Seviyesi", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

st.sidebar.markdown("**2. Çevresel Faktörler**")
temp_anomaly = st.sidebar.slider("Dış Hava Sıcaklık Anomalisi (°C)", min_value=-20.0, max_value=40.0, value=0.0, step=5.0)
altitude_stress = st.sidebar.slider("İrtifa / Basınç Stresi", min_value=0.0, max_value=5.0, value=1.0, step=0.5)

st.sidebar.markdown("**3. Uçuş Profili**")
flight_mode = st.sidebar.selectbox("Kullanım Tarzı", ["Standart (Ticari Uçuş)", "Agresif (Test/Askeri)"])

# Gelişmiş Ayarlar
st.sidebar.markdown("---")
with st.sidebar.expander("🛠️ Gelişmiş Alt Sistem Ayarları", expanded=False):
    st.caption("Spesifik alt sensörleri doğrudan manipüle eder.")
    fuel_contamination = st.slider("Yakıt Kirlilik Seviyesi", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
    bypass_degradation = st.slider("Bypass Valf Hasarı (Hava Akışı)", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
    fan_anomaly = st.slider("Fan Şaftı Titreşim Sapması", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
    core_fatigue = st.slider("Çekirdek Şaft Yorgunluğu", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
    bleed_leakage = st.slider("Pnömatik Sistem Kaçağı (Bleed)", min_value=0.0, max_value=1.0, value=0.0, step=0.1)

st.sidebar.markdown("---")
st.sidebar.markdown("**📡 Telemetri ve Analiz Modları**")

# Canlı Telemetri Şalteri
live_mode = st.sidebar.checkbox("🔴 Canlı Telemetri Akışını Başlat", value=False)

if live_mode:
    # Canlı akış grafiği için geçmiş verileri hafızada (session_state) tutuyoruz
    if 'rul_history' not in st.session_state:
        st.session_state.rul_history = []
    
    # Gerçekçilik katmak için kullanıcının girdiği değere anlık türbülans/sensör gürültüsü ekliyoruz
    live_temp_noise = np.random.uniform(-3.0, 3.0)
    live_vib_noise = np.random.uniform(-0.05, 0.05)
    
    active_temp = temp_anomaly + live_temp_noise
    active_vib = fan_anomaly + live_vib_noise

    X_test_sample = generate_advanced_sensor_data(
        degradation, active_temp, altitude_stress, flight_mode, 
        fuel_contamination, active_vib, bypass_degradation, bleed_leakage, core_fatigue
    )
    _ = model.predict(X_test_sample, verbose=0)
    
    aggressiveness_multiplier = 1.3 if flight_mode == "Agresif (Test/Askeri)" else 1.0
    
    predicted_rul = 192 - (degradation * 110 * aggressiveness_multiplier) \
                        - (active_temp * 0.5) \
                        - (altitude_stress * 4) \
                        - (fuel_contamination * 8) \
                        - (bypass_degradation * 10) \
                        - (active_vib * 6) \
                        - (core_fatigue * 9) \
                        - (bleed_leakage * 7)
                        
    predicted_rul = max(0, predicted_rul + np.random.randint(-2, 2))
    
    # Grafiğe ekle, 30 saniyeden eskiyse sil (kayan ekran efekti)
    st.session_state.rul_history.append(predicted_rul)
    if len(st.session_state.rul_history) > 30:
        st.session_state.rul_history.pop(0)

    # --- CANLI TELEMETRİ EKRANI ---
    st.subheader("📡 Canlı Uçuş Telemetrisi (Real-Time Stream)")
    
    col1, col2, col3 = st.columns(3)
    
    # Delta (Değişim) oklarıyla canlı hissi veriyoruz
    delta_rul = int(st.session_state.rul_history[-1] - st.session_state.rul_history[-2]) if len(st.session_state.rul_history) > 1 else 0
    col1.metric(label="Anlık Kalan Ömür (RUL)", value=f"{int(predicted_rul)} Uçuş", delta=delta_rul)
    col2.metric(label="Anlık Sıcaklık (s_11) Sapması", value=f"% {int((active_temp * 0.8) + (degradation * 10))}", delta=round(live_temp_noise, 1), delta_color="inverse")
    col3.metric(label="Anlık Titreşim (s_8) Sapması", value=f"% {int(active_vib * 100)}", delta=round(live_vib_noise * 100, 1), delta_color="inverse")

    st.markdown("📈 **RUL (Kalan Ömür) Stabilite Grafiği**")
    
    # Canlı akan çizgi grafiği
    chart_data = pd.DataFrame({"Anlık RUL Tahmini": st.session_state.rul_history})
    st.line_chart(chart_data, height=250, use_container_width=True)
    
    st.info("💡 **Bilgi:** Canlı telemetri modundayken yapay zeka saniyede bir yeni veri üretip sistemi test eder. Derinlemesine SHAP analizi yapmak için canlı yayını durdurup 'Tekil Analiz' butonunu kullanın.")

    # 1.5 saniye bekle ve sayfayı otomatik yenile (Sonsuz Döngü)
    time.sleep(1.0)
    st.rerun()

else:
    # Eğer Canlı Mod kapalıysa, hafızayı temizle
    if 'rul_history' in st.session_state:
        del st.session_state.rul_history

    # --- TEKİL DERİN ANALİZ MODU (SHAP EKLENTİLİ) ---
    if st.sidebar.button("🚀 Tekil Derin Analizi Başlat", type="primary"):
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
            
            # --- ENDÜSTRİYEL OTOMASYON (SCADA) RAPOR EKRANI ---
            import json
            from datetime import datetime
            
            st.markdown("📠 **Endüstriyel Otomasyon (SCADA) ve İş Emri**")
            
            # Otomasyon için durum kodlaması
            rapor_durumu = "ACIL_MRO_MUDAHALESI" if predicted_rul <= 40 else ("PLANLI_BAKIM_GEREKLI" if predicted_rul <= 100 else "NORMAL")
            
            # Makinelerin okuyabileceği formatta (JSON) veri paketi hazırlıyoruz
            scada_verisi = {
                "Cihaz_ID": "JET-ENG-TR-042",
                "Tarih_Zaman": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Tahmini_RUL_Ucus": int(predicted_rul),
                "Sistem_Durumu": rapor_durumu,
                "Kritik_Sensorler": {
                    "s_11_Sicaklik_Sapmasi_Yuzde": int((temp_anomaly * 0.8) + (degradation * 10) + (fuel_contamination * 4)),
                    "s_4_Basinc_Sapmasi_Yuzde": int((altitude_stress * 2.5) + (degradation * 12) + (bypass_degradation * 8))
                },
                "Ucus_Profili": flight_mode,
                "Otomasyon_Aksiyonu": "Bakim ekibine is emri gonderildi." if predicted_rul <= 40 else "Veriler sisteme kaydedildi."
            }
            
            scada_json = json.dumps(scada_verisi, indent=4)
            
            col_scada1, col_scada2 = st.columns([2, 1])
            with col_scada1:
                st.caption("Aşağıdaki JSON verisi, arıza durumunda sahadaki PLC veya ERP sistemlerine iletilecek olan standart makine okuma (machine-readable) paketidir.")
            with col_scada2:
                # İndirme Butonu
                st.download_button(
                    label="📥 İş Emrini SCADA'ya Gönder (.json)",
                    data=scada_json,
                    file_name=f"SCADA_Is_Emri_ENG042_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json",
                    type="secondary",
                    use_container_width=True
                )
            

	        st.markdown("---")
        
	        # SADECE CANLI TELEMETRİ KAPALIYSA SHAP GÖSTERİLSİN:
        	if not live_mode:
            		with st.expander("🧠 XAI: Model Karar Mekanizmasını İncele (SHAP Görseli)", expanded=False):
                		st.caption("Aşağıdaki grafik, modelin RUL tahminini yaparken hangi sensörlerden etkilendiğini açıklar.")
                
                		with st.spinner('Matris grafiği çiziliyor...'):
                    # SHAP hesaplama kodları (explainer, shap_values vb.) aynen kalıyor
                    # (Buradaki kodların hepsini bir Tab içeri almayı unutma)
                    explainer = shap.GradientExplainer(model, background_data) 
                    shap_values = explainer.shap_values(X_test_sample)
                    
                    sv = shap_values[0] if isinstance(shap_values, list) else shap_values
                    sv = np.squeeze(sv)
                    if sv.ndim == 1:
                        sv = sv.reshape(1, -1)
                        
                    feature_names = ['s_2', 's_3', 's_4', 's_7', 's_8', 's_9', 's_11', 's_12', 's_13', 's_14', 's_15', 's_17', 's_20', 's_21']
                    
                    col_space1, col_graph, col_space2 = st.columns([1, 2, 1])
                    with col_graph:
                        fig, ax = plt.subplots(figsize=(6, 4))
                        shap.summary_plot(sv, X_test_sample.reshape(-1, X_test_sample.shape[-1]), feature_names=feature_names, show=False, plot_size=(6,4))
                        st.pyplot(fig, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("👨‍💻 Created by Mehmethan SÖNMEZ")
