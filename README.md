# Corteva Seed Demand Prediction

This project predicts product demand using machine learning and generative AI tools.

## Problem Statement

Optimize inventory and sales strategies by forecasting demand for a diverse product portfolio across multiple states, using product traits and historical sales data.

## Approach

- Exploratory Data Analysis (EDA) and feature engineering
- ML modeling (regression, hybrid, XGBoost)
- (GenAI data enrichment and scenario simulation (not implemented))
- Streamlit web app frontend
- (Deployment on Azure (Web App, Blob Storage, DevOps) (not implemented))

## Quick Start

1. Clone the repo:
git clone
2. Create the environment from the YAML file:
conda env create -f environment.yml
3. Activate the environment (I forgot to change the env name before pip freeze, so the name is not appropriate here):
conda activate dsci510
4. Run the app:
streamlit run app/app.py

## Repo Structure

- `data/`: Sample data for reproducibility (no file here)
- `notebooks/`: Jupyter notebooks for EDA and modeling
- `app/`: Web app code (Streamlit)
- `README.md`: Project overview and instructions