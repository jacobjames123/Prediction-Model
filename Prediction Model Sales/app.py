import streamlit as st
import pandas as pd
import pickle

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('columns.pkl', 'rb') as file:
    expected_columns = pickle.load(file)

st.title("Sticker Sales Predictor")

country = st.selectbox("Country", ["Canada", "Finland", "Italy", "Kenya", "Norway", "Singapore"])
store = st.selectbox("Store", ["Discount Stickers", "Premium Sticker Mart", "Stickers for Less"])
product = st.selectbox("Product", ["Kaggle", "Kaggle Tiers", "Kerneler", "Kerneler Dark Mode"])
date = st.date_input("Date")

def preprocess_input(country, store, product, date):
    data = pd.DataFrame({
        'country': [country],
        'store': [store],
        'product': [product],
        'day_of_week': [date.weekday()],  
        'month': [date.month]
    })
    data_encoded = pd.get_dummies(data, drop_first=True)
    data_encoded = data_encoded.reindex(columns=expected_columns, fill_value=0)
    return data_encoded


if st.button("Predict"):
    input_data = preprocess_input(country, store, product, date)
    prediction = model.predict(input_data)
    st.write(f"Predicted Sales: {prediction[0]:.2f}")