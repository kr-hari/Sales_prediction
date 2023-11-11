import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Load your sales time series data into a pandas DataFrame
# Replace 'your_data.csv' with the path to your data file
data = pd.read_csv('your_data.csv')

# Create a Streamlit web app with a title and some styling
st.title('Sales Time Series Dashboard')
st.markdown(
    """
    This dashboard allows you to select a store and view smoothed sales data along with LSTM predictions.
    """
)
st.sidebar.markdown("### Select Options")

# Add space for better layout
st.sidebar.write("")
st.sidebar.write("")

# Filter by store
selected_store = st.sidebar.selectbox('Select Store:', data['Store'].unique())

# Filter the data by the selected store
filtered_data = data[data['Store'] == selected_store]

# Smooth the data using a moving average
smoothed_data = filtered_data['Sales'].rolling(window=7).mean()

# Create a DataFrame for the smoothed data
smoothed_df = pd.DataFrame({'Date': filtered_data['Date'], 'Smoothed Sales': smoothed_data})

# Create an Altair line chart for the smoothed data
chart = alt.Chart(smoothed_df).mark_line().encode(
    x='Date:T',
    y='Smoothed Sales:Q',
).properties(
    width=800,
    height=400
)

# Display the chart using Streamlit
st.altair_chart(chart, use_container_width=True)

# Add a refresh button to trigger LSTM prediction
if st.sidebar.button('Refresh and Make Predictions'):
    # Extract the smoothed sales data for prediction
    smoothed_sales = smoothed_data.dropna().values.reshape(-1, 1)
    
    # Normalize the data for LSTM
    scaler = MinMaxScaler()
    scaled_sales = scaler.fit_transform(smoothed_sales)
    
    # Prepare data for LSTM (input sequences and corresponding labels)
    look_back = 10  # You can adjust this value for your model
    X, y = [], []
    for i in range(len(scaled_sales) - look_back):
        X.append(scaled_sales[i:i+look_back, 0])
        y.append(scaled_sales[i+look_back, 0])
    X, y = np.array(X), np.array(y)
    
    # Reshape the input for LSTM (samples, time steps, features)
    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
    
    # Create and train an LSTM model
    model = Sequential()
    model.add(LSTM(50, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, y, epochs=100, batch_size=1, verbose=2)
    
    # Make predictions using the trained model
    test_input = scaled_sales[-look_back:].reshape(1, 1, look_back)
    predicted_sales = model.predict(test_input)
    
    # Inverse transform the predictions to the original scale
    predicted_sales = scaler.inverse_transform(predicted_sales)
    
    # Create a DataFrame for the predicted data
    predicted_df = pd.DataFrame({'Date': smoothed_df['Date'].iloc[-1:], 'Predicted Sales': predicted_sales.flatten()})
    
    # Create a chart for the predicted data
    prediction_chart = alt.Chart(predicted_df).mark_line(color='red').encode(
        x='Date:T',
        y='Predicted Sales:Q',
    ).properties(
        width=800,
        height=400
    )
    
    # Display the predicted data in a beautiful way
    st.sidebar.write("")
    st.sidebar.markdown("### Predicted Sales")
    st.sidebar.altair_chart(prediction_chart, use_container_width=True)