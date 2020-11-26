import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pycaret.regression import *
import base64
import io
import requests


# 定義 Download Function
def download_link(object_to_download, download_filename, download_link_text):

    if isinstance(object_to_download,pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(object_to_download.encode()).decode()

    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'


# 定義 Background
page_bg_img = '''
<style>
body {
background-image: url("https://images.unsplash.com/photo-1493606278519-11aa9f86e40a?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1050&q=80");
background-size: cover;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)


# App Title
st.write("""
# Price Prediction App
This app predicts the **house price**

Data obtained from [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
""")


### Default File
url = "https://raw.githubusercontent.com/Chiehcode/House_Price_Web_App/main/default.csv"

# Collects user input features into dataframe
uploaded_file = st.file_uploader(
    "Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    input_df = pd.read_csv(url, error_bad_lines=False)


# Apply Download Function for Sample Data

url_data = requests.get(url).content
sample_data = pd.read_csv(io.StringIO(url_data.decode('utf-8')))

if st.button('下載範例格式'):
    tmp_download_link = download_link(sample_data, 'Sample.csv', 'Click here to download Sample Data')
    st.markdown(tmp_download_link, unsafe_allow_html=True)


# Displays the user input features
st.subheader('特徵值')

if uploaded_file is not None:
    st.write(input_df)
else:
    st.write(
        'Sample')
    st.write(input_df)


# load the model
saved_final_cb = load_model('Final CatBoost Model')


# Apply model to make predictions
prediction = saved_final_cb.predict(input_df)

st.subheader('預測結果')

df = pd.DataFrame({
    'Prediction': prediction,
})

st.write(df)


# Apply Download Function for Predition Data
if st.button('下載預測結果'):
    tmp_download_link = download_link(df, 'Prediction.csv', 'Click here to download Predition Data')
    st.markdown(tmp_download_link, unsafe_allow_html=True)
