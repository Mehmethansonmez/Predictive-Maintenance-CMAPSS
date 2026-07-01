<div align="center">

# ✈️ Jet Engine Predictive Maintenance & XAI (SHAP) Dashboard

**Derin Öğrenme (LSTM) ve Açıklanabilir Yapay Zeka (XAI) tabanlı Endüstriyel Kestirimci Bakım Simülatörü**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

</div>

---

## 📋 Proje Özeti (Overview)
Bu proje, **NASA CMAPSS (FD004)** veri seti üzerinde eğitilmiş, jet motorlarının Kalan Kullanım Ömrünü (RUL) tahmin eden uçtan uca bir kestirimci bakım sistemidir. 6 farklı uçuş rejimi ve hibrit arıza modlarını (kompresör + fan) içeren "Zor Mod" (FD004) verileriyle eğitilmiştir.

Sistem, **SHAP** entegrasyonu sayesinde "kara kutu" yapay zeka kararlarını şeffaflaştırır ve kritik durumlarda **SCADA/ERP sistemlerine** makine-okunabilir JSON formatında bakım iş emri üreterek MRO süreçlerine entegre olur.

## ✨ Temel Özellikler (Key Features)
* **Derin Öğrenme Mimarisi:** Gürültüden arındırılmış (EWMA + Z-Score Normalization) verilerle eğitilmiş Çift Katmanlı **LSTM** ağı.
* **Canlı Telemetri Simülatörü:** Uçuş profili ve çevresel stres faktörlerinin (sıcaklık/basınç) RUL üzerindeki anlık etkisini görselleştirme.
* **Şeffaf Yapay Zeka (XAI):** Tahminlerin temelindeki 14 kritik sensör anormalliğini vurgulayan dinamik SHAP Karar Mekanizması.
* **Endüstriyel Entegrasyon (SCADA):** Kritik RUL eşiklerinde, bakım ekipleri için PLC/ERP sistemlerine uygun .json iş emri üretme özelliği.

## 🗂️ Veri Seti ve Sensörler
Sistem, gürültü eleme aşamasından geçen **14 kritik termodinamik ve fiziksel sensörü** (s_2, s_3, s_4, s_7, s_8, s_9, s_11, s_12, s_13, s_14, s_15, s_17, s_20, s_21) kullanmaktadır.
* **Rejim Normalizasyonu:** K-Means algoritması ile 6 farklı uçuş rejimi kümelenmiş, modelin irtifa gürültüsünden etkilenmemesi sağlanmıştır.

## 🛠️ Kullanılan Teknolojiler
* **Yapay Zeka:** `TensorFlow/Keras (LSTM)`, `scikit-learn (K-Means)`
* **Açıklanabilirlik:** `SHAP` (GradientExplainer)
* **Web & Arayüz:** `Streamlit`, `Matplotlib`, `Pandas`
* **Otomasyon:** `JSON-based SCADA Integration`

## 🚀 Kurulum ve Kullanım
NOT: KURULUM YAPMADAN DOĞRUDAN KULLANMAK İSTERSENİZ SAYFADA BULUNAN LİNK ÜZERİNDEN DE ERİŞEBİLİRSİNİZ.
```bash
# 1. Repoyu Klonlayın
git clone [https://github.com/Mehmethansonmez/Predictive-Maintenance-CMAPSS.git](https://github.com/Mehmethansonmez/Predictive-Maintenance-CMAPSS.git)
cd Predictive-Maintenance-CMAPSS

# 2. Gereksinimleri Yükleyin
pip install -r requirements.txt

# 3. Uygulamayı Başlatın
streamlit run app.py
