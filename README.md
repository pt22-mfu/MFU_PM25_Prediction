# MFU PM2.5 Forecasting Engine
**Developed by:** The Outliers (Computer Engineering, MFU)
**Advisor:** Aj. Khwunta Kirimasthong

## 📌 Project Overview
This project is part of the **Engineering Pre-project** course. It aims to provide high-accuracy, real-time PM2.5 forecasting specifically for the **Mae Fah Luang University (MFU) Valley** area using Machine Learning.

## 🛠️ Tech Stack
* **Language:** Python 3.10+
* **Framework:** Streamlit (Web UI)
* **ML Architecture:** XGBoost, Random Forest, SVR, MLR
* **Data Sources:** OpenWeatherMap API, NASA FIRMS (Spatial Data), Local Historical CSV

## 📂 Repository Structure
* `app_v2.py`: Main dashboard application.
* `pm25_model.pkl`: Trained XGBoost model for prediction.
* `fetch_firedata_NASA.py`: Live spatial data acquisition script.
* `etl_pipeline.py`: Data cleaning and feature engineering pipeline.
* `pm25_data.csv`: Historical dataset used for model baseline.

## 🚀 Key Features
* **Multi-Model Comparison:** Real-time performance tracking between XGBoost and baseline models.
* **Spatial Feature Integration:** Incorporating regional environmental factors into the prediction logic.
* **Live Forecast:** 5-day predictive trend analysis for the MFU campus.

---
PT
