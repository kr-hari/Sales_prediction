import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# Load your sales time series data into a pandas DataFrame
# Replace 'your_data.csv' with the path to your data file
data = pd.read_csv('data/rossmann-store-sales/train.csv')

# Create a Streamlit web app
st.title('Sales Time Series Dashboard')

# Filter by store
selected_store = st.selectbox('Select Store:', data['Store'].unique())

# Filter the data by the selected store
filtered_data = data[data['Store'] == selected_store]

# Smooth the data using a moving average
smoothed_data = filtered_data['Sales'].rolling(window=7).mean()

# Create a DataFrame for the smoothed data
smoothed_df = pd.DataFrame({'Date': filtered_data['Date'], 'Smoothed Sales': smoothed_data})

# Create an Altair line chart
chart = alt.Chart(smoothed_df).mark_line().encode(
    x='Date:T',
    y='Smoothed Sales:Q',
).properties(
    width=800,
    height=400
)

# Display the chart using Streamlit
st.altair_chart(chart, use_container_width=True)