# 🌫️ Real-Time PM2.5 Forecasting Engine for Northern Thailand

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-XGBoost-orange.svg)
![Data Engineering](https://img.shields.io/badge/Data%20Pipeline-ETL-success.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)

**Developed by:** The Outliers (Computer Engineering, MFU)  
**Advisor:** Aj. Khwunta Kirimasthong  

## 📌 Project Overview
Northern Thailand frequently experiences severe air quality degradation due to seasonal agricultural burning and microclimate effects. Traditional weather models often fail to predict PM2.5 spikes in localized valleys accurately. 

This project is a complete end-to-end Machine Learning pipeline designed to provide high-accuracy, real-time PM2.5 forecasting specifically for the **Mae Fah Luang University (MFU) Valley**. By integrating real-time meteorological data with spatial fire hotspot data (NASA FIRMS), this engine provides a proactive 5-day predictive trend to aid public health decisions.

## 🏗️ System Architecture & Data Pipeline
The core strength of this project lies in its automated ETL process and robust feature engineering:

1. **Extraction:** Real-time data fetching from **OpenWeatherMap API** (meteorological conditions) and **NASA FIRMS API** (spatial active fire data/hotspots).
2. **Transformation (`etl_pipeline.py`):** * Automated data cleaning and handling of missing values.
   * Spatial feature engineering (calculating the impact of nearby fire hotspots based on wind direction and distance).
   * Time-series lagging features generation.
3. **Modeling:** Making predictions using the tuned XGBoost algorithm.
4. **Serving:** Real-time visualization and forecasting via an interactive Streamlit Web Dashboard.

## 🧠 Machine Learning Performance
We evaluated multiple models (XGBoost, Random Forest, SVR, Multiple Linear Regression). **XGBoost** outperformed the baselines by capturing non-linear relationships between wind patterns and fire hotspots.

* **Primary Model:** XGBoost Regressor
* **R² Score:** [Insert R2 Score, e.g., 0.85]
* **RMSE:** [Insert RMSE value] µg/m³
* *Note: The model successfully reduces prediction latency while maintaining high accuracy during peak burning seasons.*

## 🛠️ Tech Stack
* **Data Engineering:** Python (Pandas, NumPy), RESTful API Integration, Automated ETL workflows.
* **Machine Learning:** Scikit-Learn, XGBoost.
* **Deployment & UI:** Streamlit.
* **Data Sources:** OpenWeatherMap API, NASA FIRMS, Local Historical Data.

## 📂 Repository Structure
```text
├── app_v2.py                 # Main Streamlit dashboard application for live forecasting
├── pm25_model.pkl            # Serialized/Trained XGBoost model for rapid inference
├── fetch_firedata_NASA.py    # Automated script for live spatial data acquisition
├── etl_pipeline.py           # Core Data Engineering script (Cleaning & Feature Engineering)
├── pm25_data.csv             # Historical dataset (Baseline)
└── requirements.txt          # Python dependencies
