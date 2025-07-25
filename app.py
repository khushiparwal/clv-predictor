import streamlit as st
import pandas as pd
import pickle
from io import BytesIO

with open("model.pkl", "rb") as file:
    model = pickle.load(file)

st.set_page_config(page_title="CLV Segment Predictor", layout="wide")
st.title("Customer Lifetime Value (CLV) Segment Predictor")

st.markdown("Upload a customer RFM dataset in Excel format (`.xlsx`) to predict CLV segments.")

uploaded_file = st.file_uploader("Upload Excel file (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    try:
        data = pd.read_excel(uploaded_file)
        st.subheader("Preview of Uploaded Data")
        st.write(data.head())
        drop_cols = ["Invoice", "StockCode", "Description", "InvoiceDate", "Country", "LTVCluster", "m6_Revenue", "Segment","Revenue_x", "Revenue_y"]
        features = data.drop(columns=drop_cols, errors="ignore")
        expected_cols = ['CustomerID', 'Recency', 'RecencyCluster', 'Frequency', 'FrequencyCluster',
                 'Revenue', 'RevenueCluster', 'OverallScore',
                 'Segment_High-Value', 'Segment_Low-Value', 'Segment_Mid-Value']

        for col in expected_cols:
            if col not in features.columns:
                features[col] = 0  # create missing with zeros

        features = features[expected_cols] 
        st.write("Features used for prediction:", features.columns.tolist())
        features = features.select_dtypes(include=['number', 'bool'])
        preds = model.predict(features)
        data["Predicted_LTVCluster"] = preds

        st.subheader("Predicted CLV Segments")
        st.write(data)

        output = BytesIO()
        data.to_excel(output, index=False, engine='openpyxl')
        output.seek(0)

        st.download_button(
            label="Download Results as Excel",
            data=output,
            file_name="clv_predictions.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        st.error(f"Error reading Excel file: {e}")
