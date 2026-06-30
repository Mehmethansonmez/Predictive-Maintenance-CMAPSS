<div align="center">

# ✈️ Jet Engine Predictive Maintenance & XAI (SHAP) Dashboard

**Derin Öğrenme (LSTM) ve Açıklanabilir Yapay Zeka (XAI) tabanlı Kestirimci Bakım Simülatörü**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

</div>

---

## 📋 Proje Özeti (Overview)
Bu proje, turbofan jet motorlarının sensör verilerini analiz ederek **Kalan Kullanım Ömrünü (RUL - Remaining Useful Life)** tahmin eden gelişmiş bir makine öğrenmesi sistemidir. 

Sistem, geleneksel "kara kutu" (black box) yapay zeka modellerinin aksine **SHAP** entegrasyonu sayesinde verdiği kararları şeffaflıkla açıklar. Geliştirilen Streamlit web arayüzü sayesinde kullanıcılar; çevresel faktörleri ve uçuş profillerini manipüle ederek modelin tepkilerini canlı olarak test edebilirler.

## ✨ Temel Özellikler (Key Features)
* **Zaman Serisi Analizi:** Motorun son 30 uçuş döngüsündeki 18 farklı sensör verisini işleyen LSTM (Long Short-Term Memory) mimarisi.
* **Canlı Uçuş Simülatörü:** Dış hava sıcaklığı, irtifa stresi ve uçuş agresifliği gibi çevresel faktörlerin anlık olarak RUL üzerindeki etkisinin simülasyonu.
* **Şeffaf Yapay Zeka (XAI):** Arıza tahminlerinin hangi sensör anormalliklerinden kaynaklandığını gösteren dinamik SHAP grafikleri.
* **Bulut Entegrasyonu:** Herhangi bir kurulum gerektirmeden 7/24 erişilebilir bulut tabanlı (Streamlit Cloud) web uygulaması.

## 🗂️ Veri Seti ve Sensörler (Dataset)
Sistem, motorun anlık durumunu belirlemek için aşağıdaki kritik parametreleri (ve 18 sensör değerini) kullanmaktadır:
* **s_11 (Sıcaklık Sensörü):** Motor içi termodinamik aşınma göstergesi.
* **s_4 (Basınç Sensörü):** İrtifa ve basınç stres göstergesi.
* **time_in_cycles:** Motorun tamamladığı operasyonel uçuş döngüsü.
* *Not: Tahminler, geçmiş uçuş döngülerinin 3B matrisleri (30x18) üzerinden yapılmaktadır.*

## 🛠️ Kullanılan Teknolojiler (Technologies Used)
* **Yapay Zeka:** `TensorFlow`, `Keras` (LSTM)
* **XAI (Açıklanabilirlik):** `SHAP`
* **Veri Analizi:** `NumPy`, `Pandas`
* **Görselleştirme & Web Arayüzü:** `Streamlit`, `Matplotlib`

## 🚀 Kurulum ve Kullanım (Installation & Setup)
Projeyi kendi lokal ortamınızda (bilgisayarınızda) çalıştırmak isterseniz aşağıdaki komutları tek seferde kopyalayıp terminalinize yapıştırabilirsiniz:

NOT: KURULUM YAPMADAN DOĞRUDAN KULLANMAK İSTERSENİZ SAYFADA BULUNAN LİNK ÜZERİNDEN DE ERİŞEBİLİRSİNİZ.
```bash
# 1. Repoyu Klonlayın ve Klasöre Girin
git clone [https://github.com/Mehmethansonmez/Predictive-Maintenance-CMAPSS.git](https://github.com/Mehmethansonmez/Predictive-Maintenance-CMAPSS.git)
cd Predictive-Maintenance-CMAPSS

# 2. Gerekli Kütüphaneleri Yükleyin (Python 3.11 önerilir)
pip install -r requirements.txt

# 3. Uygulamayı Başlatın
streamlit run app.py
