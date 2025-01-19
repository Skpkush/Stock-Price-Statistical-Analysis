import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import plotly.graph_objs as go
from scipy.stats import skew, kurtosis, shapiro
from arch import arch_model


# Function to fetch stock data
@st.cache
def fetch_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    return data


# Function to calculate rolling volatility
def rolling_volatility(data, window=30):
    return data['Close'].pct_change().rolling(window=window).std()


# Function to perform ADF test for stationarity
def adf_test(series):
    result = adfuller(series)
    return result[1]  # Return p-value for stationarity


# Function to calculate financial ratios
def calculate_financial_ratios(data):
    daily_returns = data['Close'].pct_change()
    return {
        'Mean Return': daily_returns.mean(),
        'Std Dev of Return': daily_returns.std(),
        'Skewness': skew(daily_returns.dropna()),
        'Kurtosis': kurtosis(daily_returns.dropna())
    }


# Main Streamlit application
def app():
    # Streamlit widgets for user input
    st.title('Stock Price Statistical Analysis')
    ticker = st.text_input('Enter Stock Ticker Symbol (e.g., AAPL, MSFT, TSLA)', 'AAPL')
    start_date = st.date_input('Start Date', pd.to_datetime('2020-01-01'))
    end_date = st.date_input('End Date', pd.to_datetime('2023-01-01'))

    # Fetch data
    data = fetch_data(ticker, start_date, end_date)

    if data.empty:
        st.error("No data found for the given stock symbol.")
        return

    st.subheader(f'Data for {ticker} from {start_date} to {end_date}')
    st.write(data.tail())

    # Distribution Analysis
    st.subheader('Distribution Analysis')
    st.write("Histogram of stock closing prices")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data['Close'], kde=True, ax=ax)
    st.pyplot(fig)

    # Normality Test (Shapiro-Wilk test)
    stat, p_value = shapiro(data['Close'].dropna())
    st.write(f"Shapiro-Wilk Normality Test: Stat={stat}, p-value={p_value}")
    if p_value > 0.05:
        st.write("Data seems to follow a normal distribution.")
    else:
        st.write("Data does not follow a normal distribution.")

    # Time Series Analysis
    st.subheader('Time Series Analysis')

    # ACF and PACF Plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    plot_acf(data['Close'], ax=ax1)
    plot_pacf(data['Close'], ax=ax2)
    st.pyplot(fig)

    # ADF Test for Stationarity
    p_value = adf_test(data['Close'])
    st.write(f"ADF Test p-value: {p_value}")
    if p_value < 0.05:
        st.write("The time series is stationary.")
    else:
        st.write("The time series is not stationary.")

    # ARIMA Model Forecasting
    st.subheader('ARIMA Model Forecasting')
    model = ARIMA(data['Close'], order=(5, 1, 0))
    model_fit = model.fit()

    # Reduce font size for model summary
    arima_summary_html = f"<pre style='font-size:12px;'>{model_fit.summary()}</pre>"
    st.markdown(arima_summary_html, unsafe_allow_html=True)

    forecast_steps = 30
    forecast = model_fit.forecast(steps=forecast_steps)
    st.write(f"Forecast for next {forecast_steps} days: {forecast}")

    # Plot the forecast
    forecast_dates = pd.date_range(data.index[-1], periods=forecast_steps + 1, freq='B')[1:]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data.index, data['Close'], label='Historical Data')
    ax.plot(forecast_dates, forecast, label='Forecasted Data', linestyle='--')
    ax.legend()
    st.pyplot(fig)

    # Volatility Analysis
    st.subheader('Volatility Analysis')
    vol = rolling_volatility(data)
    st.write(f"Rolling Volatility (30 days):")
    st.line_chart(vol)

    # GARCH Model for Volatility Forecasting
    st.subheader('GARCH Model')
    model = arch_model(data['Close'].pct_change().dropna(), vol='Garch', p=1, q=1)
    model_fit = model.fit(disp="off")

    # Reduce font size for GARCH model summary
    garch_summary_html = f"<pre style='font-size:12px;'>{model_fit.summary()}</pre>"
    st.markdown(garch_summary_html, unsafe_allow_html=True)

    # Financial Ratios and Indicators
    st.subheader('Financial Ratios and Indicators')
    ratios = calculate_financial_ratios(data)
    st.write(ratios)

if __name__ == "__main__":
    app()
