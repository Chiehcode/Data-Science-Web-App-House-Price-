import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
from pycaret.regression import *


# page_bg_img = '''
# <style>
# body {
# background-image: url("https://images.unsplash.com/photo-1601662528567-526cd06f6582?ixlib=rb-1.2.1&auto=format&fit=crop&w=658&q=80");
# background-size: cover;
# }
# </style>
# '''
# st.markdown(page_bg_img, unsafe_allow_html=True)



st.write("""
# Price Prediction App
This app predicts the **house price**
""")


# Collects user input features into dataframe
uploaded_file = st.file_uploader(
    "Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    input_df = pd.read_csv("C:\桌面\Python\PyCaret & Streamlit\house-prices-advanced-regression-techniques\default.csv")

st.markdown("""
[CSV 檔範例格式](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
""")


# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(input_df)
else:
    st.write(
        '範例')
    st.write(input_df)


# load the model
saved_final_cb = load_model('Final CatBoost Model')


# Apply model to make predictions
prediction = saved_final_cb.predict(input_df)


st.subheader('預測結果')
st.write(prediction)


# Explaining the model's predictions using SHAP values
# https://github.com/slundberg/shap
# explainer = shap.TreeExplainer(saved_final_cb)
# shap_values = explainer.shap_values(input_df)

# st.header('Feature Importance')
# plt.title('Feature importance based on SHAP values')
# shap.summary_plot(shap_values, input_df)
# st.pyplot(bbox_inches='tight')
# st.write('---')

# plt.title('Feature importance based on SHAP values (Bar)')
# shap.summary_plot(shap_values, input_df, plot_type="bar")
# st.pyplot(bbox_inches='tight')