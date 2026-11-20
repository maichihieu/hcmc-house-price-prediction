# HCMC House Price Prediction

> **IE224 – Data Analysis Course Project**  
> Predict residential real estate prices in Ho Chi Minh City using machine learning.

---

## Project Overview

This project builds an end-to-end data pipeline and desktop application to predict house prices in Ho Chi Minh City (HCMC), Vietnam.

**Key features:**
- **Web scraping** – Automated data collection from real estate listing sites
- **Data preprocessing** – Cleaning, feature engineering, handling outliers
- **Exploratory Data Analysis** – Distribution analysis, correlation heatmaps, geospatial insights
- **ML Ensemble** – CatBoost + XGBoost ensemble with log-price target
- **Desktop App** – CustomTkinter GUI with real-time prediction

---

## Project Structure
```
hcmc-house-price-prediction/
│
├── notebooks/
│   ├── 01_data_scraping.ipynb           # Web scraping pipeline
│   ├── 02_data_preprocessing.ipynb      # Data cleaning & feature engineering
│   └── 03_data_analysis_and_modeling.ipynb  # EDA, model training & evaluation
│
├── app/
│   ├── app.py                           # Main desktop application
│   ├── ctk_scrollable_dropdown.py       # Custom dropdown widget
│   └── districts_wards.json             # HCMC ward/district list
│
├── data/
│   ├── raw/data_raw.csv                 # Raw scraped data
│   └── processed/data_clean.csv        # Cleaned & engineered data
│
├── models/
│   ├── catboost_pipeline.pkl            # Trained CatBoost pipeline
│   └── xgboost_pipeline.pkl            # Trained XGBoost pipeline
│
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.9 – 3.11 (recommended)
- pip

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/hcmc-house-price-prediction.git
cd hcmc-house-price-prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash
python app/app.py
```

Pre-trained models and data files are already included in the repository, no extra setup needed.

---

## Application Usage

Fill in the following fields in the sidebar:

| Field | Description |
|---|---|
| Area (m²) | Total floor area of the property |
| Bedrooms | Number of bedrooms |
| Floors | Number of floors |
| Road Width (m) | Width of the access road |
| Frontage (m) | Width of the property's street frontage |
| Ward / District | Type to search or select from the dropdown list |

Click **PREDICT PRICE**. The app runs both models and displays the ensemble average price.

---

## Models

| Model | Description |
|---|---|
| **CatBoost** | Gradient boosting with native categorical support |
| **XGBoost** | Extreme gradient boosting |
| **Ensemble** | Simple average of both model outputs (after inverse log transform) |

The target variable is `log(price + 1)` to handle price skewness. Predictions are inverse-transformed with `expm1`.

---

## Key Libraries

- [`scikit-learn`](https://scikit-learn.org/) – Pipelines, transformers
- [`catboost`](https://catboost.ai/) – CatBoost regressor
- [`xgboost`](https://xgboost.readthedocs.io/) – XGBoost regressor
- [`customtkinter`](https://github.com/TomSchimansky/CustomTkinter) – Modern Tkinter GUI
- [`pandas`](https://pandas.pydata.org/) / [`numpy`](https://numpy.org/) – Data manipulation

