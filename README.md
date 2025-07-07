Customer Lifetime Value (CLV) Segment Predictor
This Streamlit web application predicts Customer Lifetime Value (CLV) segments (Low, Mid, High) based on RFM features—Recency, Frequency, and Monetary value—using a trained XGBoost classification model.
Live App Link
https://clv-predictor-9ib5xwjjsdf2wv8hteqcqd.streamlit.app/
________________________________________
Overview
This application allows users to:
•	Upload a raw Excel file containing customer transaction data
•	Automatically compute key RFM features:
o	Recency: Days since the customer’s last purchase
o	Frequency: Total number of transactions
o	Revenue: Total monetary value of transactions
•	Perform clustering based on RFM scores
•	Classify customers into CLV segments using a machine learning model
•	Download the results in Excel format
________________________________________
Input Requirements
The input must be an Excel file (.xlsx) containing the following columns (typical of e-commerce retail datasets):
•	Invoice

•	StockCode

•	Description

•	Quantity

•	Price

•	InvoiceDate

•	Customer ID

•	Country
A sample file (adv_merge_output.xlsx) is available for reference or testing.
________________________________________
Technologies Used
•	Python

•	Streamlit

•	pandas

•	scikit-learn

•	XGBoost

•	Streamlit Cloud for deployment
________________________________________
Output
The application returns a processed dataset with an additional column named Predicted_LTVCluster, indicating the customer’s segment. Users can preview and download the results.
________________________________________
Author
Khushi Parwal
Contact : parwalkhushi@gmail.com
