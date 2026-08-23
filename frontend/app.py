"""Streamlit UI for the SuperKart Sales Forecasting model."""
import os

import pandas as pd
import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://superkart-backend:7860")

st.set_page_config(page_title="SuperKart Sales Forecast", page_icon="🛒", layout="centered")

st.title("🛒 SuperKart Sales Forecasting")
st.caption(
    "Predict the total revenue of a product in a given store. "
    "Use **Online** for a single row or **Batch** for a whole CSV."
)

tab_online, tab_batch = st.tabs(["Online Prediction", "Batch Prediction"])

with tab_online:
    st.subheader("Product & store details")
    col_l, col_r = st.columns(2)

    with col_l:
        product_weight = st.number_input("Product Weight", 0.0, 50.0, 12.66, 0.01)
        product_mrp = st.number_input("Product MRP", 0.0, 500.0, 117.08, 0.01)
        product_allocated_area = st.number_input(
            "Product Allocated Area", 0.0, 1.0, 0.027, 0.001, format="%.3f"
        )
        product_sugar_content = st.selectbox(
            "Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"]
        )
        product_id_char = st.selectbox(
            "Product Category Code", ["FD", "DR", "NC"],
            help="FD - Food, DR - Drinks, NC - Non-Consumable",
        )

    with col_r:
        product_type_category = st.selectbox(
            "Product Type Category", ["Perishables", "Non Perishables"]
        )
        store_size = st.selectbox("Store Size", ["Small", "Medium", "High"])
        store_location_city_type = st.selectbox(
            "Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"]
        )
        store_type = st.selectbox(
            "Store Type",
            ["Departmental Store", "Supermarket Type1", "Supermarket Type2", "Food Mart"],
        )
        store_age_years = st.number_input("Store Age (years)", 0, 100, 16, 1)

    payload = {
        "Product_Weight": product_weight,
        "Product_Sugar_Content": product_sugar_content,
        "Product_Allocated_Area": product_allocated_area,
        "Product_MRP": product_mrp,
        "Store_Size": store_size,
        "Store_Location_City_Type": store_location_city_type,
        "Store_Type": store_type,
        "Product_Id_char": product_id_char,
        "Store_Age_Years": store_age_years,
        "Product_Type_Category": product_type_category,
    }

    if st.button("Predict Sales", type="primary"):
        try:
            resp = requests.post(f"{BACKEND_URL}/v1/predict", json=payload, timeout=30)
            if resp.status_code == 200:
                st.success(f"Predicted sales revenue: **{resp.json()['Predicted sales']:,.2f}**")
            else:
                st.error(f"Backend returned HTTP {resp.status_code}: {resp.text}")
        except Exception as exc:
            st.error(f"Could not reach the backend API at {BACKEND_URL}: {exc}")

with tab_batch:
    st.subheader("Upload a CSV for batch prediction")
    st.caption(
        "Required columns: Product_Weight, Product_Sugar_Content, Product_Allocated_Area, "
        "Product_MRP, Store_Size, Store_Location_City_Type, Store_Type, Product_Id_char, "
        "Store_Age_Years, Product_Type_Category."
    )

    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded is not None:
        preview_df = pd.read_csv(uploaded)
        st.write("Preview:")
        st.dataframe(preview_df.head())

        if st.button("Run Batch Prediction", type="primary"):
            try:
                uploaded.seek(0)
                resp = requests.post(
                    f"{BACKEND_URL}/v1/predictbatch",
                    files={"file": uploaded.getvalue()},
                    timeout=120,
                )
                if resp.status_code == 200:
                    preds = pd.Series(resp.json()).astype(float)
                    preds.index = preds.index.astype(int)
                    preview_df["Predicted_Sales"] = preds.sort_index().values
                    st.success("Batch prediction complete.")
                    st.dataframe(preview_df)
                    st.download_button(
                        "Download predictions as CSV",
                        data=preview_df.to_csv(index=False).encode("utf-8"),
                        file_name="superkart_predictions.csv",
                        mime="text/csv",
                    )
                else:
                    st.error(f"Backend returned HTTP {resp.status_code}: {resp.text}")
            except Exception as exc:
                st.error(f"Could not reach the backend API at {BACKEND_URL}: {exc}")
