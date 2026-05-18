# 🌫️ Real-Time PM2.5 Forecasting Engine for Northern Thailand

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-XGBoost-orange.svg)
![Data Engineering](https://img.shields.io/badge/Data%20Pipeline-ETL-success.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-89%25-success.svg)

**Developed by:** The Outliers (Computer Engineering, MFU)  
**Advisor:** Aj. Khwunta Kirimasthong  
**Live Dashboard:** [https://mfu-pm25-prediction.streamlit.app/](https://mfu-pm25-prediction.streamlit.app/)

## 📌 Project Overview
Northern Thailand frequently experiences severe air quality degradation due to seasonal agricultural burning and microclimate effects. Traditional weather models often fail to predict PM2.5 spikes in localized valleys accurately. 

This project is a complete end-to-end Machine Learning pipeline designed to provide high-accuracy, real-time PM2.5 forecasting specifically for the **Mae Fah Luang University (MFU) Valley** (lat 20.045, lon 99.895). By integrating real-time meteorological data with spatial fire hotspot data (NASA FIRMS), this engine feeds a trained XGBoost model to present a proactive 5-day forecast via an interactive Streamlit web dashboard.

## 🏗️ System Architecture & Data Pipeline
The core strength of this project lies in its automated ETL process and robust feature engineering:

1. **Extraction:** Real-time data fetching from **OpenWeatherMap API** (meteorological conditions) and **NASA FIRMS API** (spatial active fire data/hotspots).
2. **Transformation (`etl_pipeline.py`):** * Automated data cleaning and handling of missing values.
   * Spatial feature engineering (calculating the impact of nearby fire hotspots based on wind direction and distance).
   * Time-series lagging features generation.
3. **Modeling:** Making predictions using the tuned ML algorithms.
4. **Serving:** Real-time visualization and multi-model comparison via Streamlit.

## 🧠 Machine Learning Performance
We evaluated multiple models (XGBoost, Random Forest, SVR, Multiple Linear Regression). As demonstrated in our Live Dashboard (Tab 3), **XGBoost** outperformed the baselines by capturing non-linear relationships between wind patterns and fire hotspots.

* **Primary Model:** XGBoost Regressor
* **Accuracy:** 89% (Deployed Version)
* *Note: The system supports real-time multi-model comparison (Tab 3) allowing users to explore predictions across Random Forest, SVR, and MLR directly on the dashboard.*

## 📂 Repository Structure
```text
├── app.py                          # v1 fully-featured dashboard
├── app_v2.py                       # v2 ACTIVE/DEPLOYED main dashboard
├── requirements.txt                # Full pinned dependency list (58 packages)
├── pm25_model.pkl                  # XGBoost Primary Model (Required)
├── rf_pm25_model.pkl               # Random Forest Model (Optional - Tab 3)
├── svr_pm25_model.pkl              # SVR Model (Optional - Tab 3)
🚀 How to Run Locally
1. Clone the repository & Install dependencies

Bash
git clone [https://github.com/pt22-mfu/MFU_PM25_Prediction.git](https://github.com/pt22-mfu/MFU_PM25_Prediction.git)
cd MFU_PM25_Prediction
pip install -r requirements.txt
2. Set API Keys (Important)
To fetch live data, do not hardcode keys. Create a .streamlit/secrets.toml file in the root directory:

Ini, TOML
[api_keys]
OWM_KEY = "your_openweathermap_key"
NASA_KEY = "your_nasa_firms_key"
3. Ensure Model Files are Present
Make sure pm25_model.pkl and pm25_data.csv are in the root directory (not inside a models/ folder).

4. Run the Dashboard

Bash
# Run the active deployed dashboard
streamlit run app_v2.py
This project was developed as part of the Undergraduate Student Senior Project in Computer Engineering Program at Mae Fah Luang University.
├── mlr_pm25_model.pkl              # MLR Model (Optional - Tab 3)
├── pm25_data.csv                   # Historical dataset (Required for Tab 2)
└── fetch_firedata_NASA.py          # Spatial data acquisition script
