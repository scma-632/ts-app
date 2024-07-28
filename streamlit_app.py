import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima
from sklearn.model_selection import train_test_split

# Streamlit app title
st.title('Stock Price Time Series Forecasting')

# User inputs
ticker = st.text_input('Enter stock ticker symbol', 'AAPL')
start_date = st.date_input('Start date', value=pd.to_datetime('2020-01-01'))
end_date = st.date_input('End date', value=pd.to_datetime('today'))
forecast_horizon = st.number_input('Enter forecast horizon (days)', min_value=1, value=30)

# Fetch stock data
data = yf.download(ticker, start=start_date, end=end_date)

# Pre-process data
data.reset_index(inplace=True)
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
ts_data = data['Adj Close']

# Decompose time series
decomposition = seasonal_decompose(ts_data, model='multiplicative', period=365)
st.write('Seasonal decomposition:')
st.pyplot(decomposition.plot())

# Train-test split
train, test = train_test_split(ts_data, test_size=0.2, shuffle=False)

# Model fitting
st.write('Fitting model...')
model = auto_arima(train, seasonal=True, m=12, suppress_warnings=True)

# Forecast
forecast = model.predict(n_periods=forecast_horizon)
forecast_index = pd.date_range(start=test.index[-1], periods=forecast_horizon + 1, closed='right')
forecast_series = pd.Series(forecast, index=forecast_index)

# Plot results
plt.figure(figsize=(14, 7))
plt.plot(ts_data, label='Actual')
plt.plot(forecast_series, label='Forecast')
plt.legend()
plt.title('Stock Price Forecast')
st.pyplot(plt)

st.write('Forecasted values:')
st.write(forecast_series)