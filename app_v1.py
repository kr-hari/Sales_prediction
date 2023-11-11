import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LinearRegression

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

# Add a refresh button to trigger prediction
if st.button('Refresh and Make Predictions'):
    # Dummy linear regression model for demonstration purposes
    model = LinearRegression()
    
    # Assuming you have features and labels in your dataset, you can train the model here.
    # X = ...  # Features
    # y = ...  # Labels
    # model.fit(X, y)
    
    # For demonstration, we'll just predict the smoothed data as an example
    predicted_sales = model.predict(smoothed_data.dropna().values.reshape(-1, 1))
    
    # Create a DataFrame for the predicted data
    predicted_df = pd.DataFrame({'Date': smoothed_df['Date'], 'Predicted Sales': predicted_sales})
    
    # Display the predicted data in a line chart
    prediction_chart = alt.Chart(predicted_df).mark_line(color='red').encode(
        x='Date:T',
        y='Predicted Sales:Q',
    ).properties(
        width=800,
        height=400
    )
    
    st.altair_chart(prediction_chart, use_container_width=True)