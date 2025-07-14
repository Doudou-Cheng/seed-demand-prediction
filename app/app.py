import streamlit as st
import joblib
import pandas as pd
import json
import numpy as np
import os

# Paths
model_path = os.path.join(os.path.dirname(__file__), 'model', 'corteva_sales_prediction_pipeline.pkl')
dict_path = os.path.join(os.path.dirname(__file__), 'model', 'product_dict.json')

pipeline = joblib.load(model_path)

# Load historical data for lag lookup
@st.cache_data
def load_database():
    db = pd.read_csv('case_study_data.csv')
    # Ensure consistent types
    db['SALESYEAR'] = db['SALESYEAR'].astype(str)
    db['PRODUCT'] = db['PRODUCT'].astype(str)
    db['STATE'] = db['STATE'].astype(str)
    return db

db = load_database()

# Load product dictionary for autofill
@st.cache_data
def load_product_dict():
    with open(dict_path, 'r') as f:
        product_dict = json.load(f)
    product_list = list(product_dict.keys())
    return product_dict, product_list

product_dict, product_list = load_product_dict()

st.title("Seed Demand Prediction")

# ---- Sidebar Navigation ----
page = st.sidebar.radio(
    "Navigation",
    ["📈 Prediction Tool", "💬 Chatbot"]
)

if page == "📈 Prediction Tool":
    # ========================
    # Manual Single Prediction
    # ========================
    st.header("🔢 Manual Prediction Input")
    product = st.selectbox("Product Name", product_list)
    prefill = product_dict.get(product, {})

    # Define defaults be minimum for each field
    defaults = {
        'RELEASE_YEAR': 1980,
        'DISEASE_RESISTANCE': 0,
        'INSECT_RESISTANCE': 0,
        'PROTECTION': 0,
        'DROUGHT_TOLERANCE': 1,
        'BRITTLE_STALK': 1,
        'PLANT_HEIGHT': 1,
        'RELATIVE_MATURITY': 1
    }

    # Impute missing values
    missing_fields = []
    for key, val in defaults.items():
        # Check for null/NaN in dictionary (JSON null loads as None; also check for np.nan)
        v = prefill.get(key)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            prefill[key] = val
            missing_fields.append(key)

    # Show warning if any missing fields were imputed
    if missing_fields:
        st.info(f"The following fields were missing for this product and have been set to the lowest/default value: {', '.join(missing_fields)}")

    # Show advanced/expandable section for manual override
    with st.expander("Advanced Options: Edit Product Features"):
        release_year = st.number_input("Release Year", min_value=1980, max_value=2100, value=int(prefill['RELEASE_YEAR']), step=1)
        disease_res = st.selectbox("Disease Resistance", [0, 1], index=int(prefill['DISEASE_RESISTANCE']))
        insect_res = st.selectbox("Insect Resistance", [0, 1], index=int(prefill['INSECT_RESISTANCE']))
        protection = st.selectbox("Protection (Above-ground)", [0, 1], index=int(prefill['PROTECTION']))
        drought_tol = st.selectbox("Drought Tolerance (1-5)", [1, 2, 3, 4, 5], index=int(prefill['DROUGHT_TOLERANCE'])-1)
        brittle_stalk = st.selectbox("Brittle Stalk (1-5)", [1, 2, 3, 4, 5], index=int(prefill['BRITTLE_STALK'])-1)
        plant_height = st.selectbox("Plant Height (1-5)", [1, 2, 3, 4, 5], index=int(prefill['PLANT_HEIGHT'])-1)
        relative_maturity = st.selectbox("Relative Maturity (1-5)", [1, 2, 3, 4, 5], index=int(prefill['RELATIVE_MATURITY'])-1)

    # The rest of the UI for non-product-based features
    state = st.selectbox("US State", ['California', 'NewYork', 'Illinois', 'Iowa', 'Texas'])
    default_salesyear = max(release_year, 2000)
    salesyear = st.number_input(
        "Sales Year", min_value=release_year, max_value=2100,
        value=default_salesyear, step=1
    )
    lifecycle = st.selectbox("Lifecycle Stage", ["Introduction", "Established", "Expansion", "Phaseout"])

    # lag1 logic
    units_lag1 = st.text_input("Previous Year's Units Sold (leave blank if unknown)") # this is an optional input cell
    # Attempt to get previous year's sales if blank
    lag_value = None
    if units_lag1.strip() == '':
        # Try to lookup from case_study_data.csv
        prior_year = int(salesyear) - 1
        # Match by PRODUCT, STATE, and SALESYEAR
        match = db[(db['PRODUCT'] == product) &
                (db['STATE'] == state) &
                (db['SALESYEAR'] == str(prior_year))]
        if not match.empty:
            lag_value = match['UNITS'].values[0]
        else:
            lag_value = None
    else:
        # Use user input (convert to float)
        try:
            lag_value = float(units_lag1)
        except:
            st.error("Invalid input for previous year's units sold. Please enter a number or leave blank.")
            lag_value = None

    # --- Prediction Starts Here ---
    if st.button("Predict Demand"):
        input_df = pd.DataFrame([{
            "PRODUCT": product,
            "SALESYEAR": pd.to_datetime(str(salesyear)),
            "LIFECYCLE": lifecycle,
            "STATE": state,
            "RELEASE_YEAR": pd.to_datetime(str(release_year)),
            "DISEASE_RESISTANCE": disease_res,
            "INSECT_RESISTANCE": insect_res,
            "PROTECTION": protection,
            "DROUGHT_TOLERANCE": drought_tol,
            "BRITTLE_STALK": brittle_stalk,
            "PLANT_HEIGHT": plant_height,
            "RELATIVE_MATURITY": relative_maturity,
            "UNITS_LAG1": lag_value
        }])

        # (Optional) If your pipeline expects lag features, handle or set as NaN
        input_df["PRODUCT_AGE"] = salesyear - release_year
        # input_df["YOY_CHANGE"] = None  

        # Predict
        pred_units = pipeline.predict(input_df)[0]
        pred_units = np.clip(pred_units, 0, 100)
        st.success(f"Predicted Units Sold: {pred_units:.1f}")
        if lag_value is None:
            st.info("Note: No previous year's sales were provided or found; prediction uses model default for lag.")


    st.markdown("---")

    # ========================
    # Batch Prediction Section
    # ========================
    st.header("📂 Batch Prediction (Upload CSV/Excel)")

    uploaded_file = st.file_uploader("Upload CSV or Excel for batch prediction", type=["csv", "xlsx"])

    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            batch_df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            batch_df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file type!")
            batch_df = None

        if batch_df is not None:
            st.write("Preview of uploaded data:", batch_df.head())

            # Make sure lag columns exist (set to None if missing)
            for col in ['UNITS_LAG1']:
                if col not in batch_df.columns:
                    batch_df[col] = None

            # Predict
            try:
                preds = pipeline.predict(batch_df)
                batch_df['Predicted_UNITS'] = preds
                st.success("Batch prediction complete!")
                st.write(batch_df.head())

                # Download results
                csv = batch_df.to_csv(index=False)
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv,
                    file_name='batch_predictions.csv',
                    mime='text/csv'
                )
            except Exception as e:
                st.error(f"Prediction failed: {e}")

elif page == "💬 Chatbot":
    st.header("🤖 Chatbot Interface")
    st.write("This is a placeholder for the chatbot functionality: ask questions about dataset, predictions (inputs formatted and sent to backend models), visualizations, etc")
    st.write("The LLM has access to product traits, historical dataset, and prediction models." \
    "")
    # Placeholder for chatbot interaction
    user_input = st.text_input("Ask a question about seed demand or predictions:")
    if user_input:
        st.write(f"You asked: {user_input}")
        st.write("Chatbot response would go here (integrate with your chatbot backend).")