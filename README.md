
# Stock Price Statistical Analysis

A Streamlit-based web application for analyzing and forecasting stock prices using statistical and machine learning techniques. This application allows users to visualize stock data, perform time series analysis, calculate financial ratios, and predict future stock movements.

---

## **Features**
1. **Fetch Stock Data:**
   - Retrieve historical stock price data using Yahoo Finance.
   - Specify the stock ticker, start date, and end date.

2. **Statistical Distribution Analysis:**
   - Histogram with a kernel density estimate (KDE) for stock prices.
   - Perform the Shapiro-Wilk test to check for data normality.

3. **Time Series Analysis:**
   - Plot Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF).
   - Perform the Augmented Dickey-Fuller (ADF) test to determine stationarity.

4. **ARIMA Model Forecasting:**
   - Fit an ARIMA model to the stock price data.
   - Generate forecasts for the next 30 business days.
   - Visualize historical and forecasted stock prices.

5. **Volatility Analysis:**
   - Calculate and display rolling volatility over a 30-day window.
   - Fit a GARCH model to predict future volatility.

6. **Financial Ratios and Indicators:**
   - Calculate mean returns, standard deviation, skewness, and kurtosis.

---

## **Technologies Used**
- **Python Libraries:**
  - `streamlit`: Web application framework.
  - `yfinance`: Fetching stock price data.
  - `numpy`, `pandas`: Data manipulation.
  - `matplotlib`, `seaborn`: Visualization.
  - `statsmodels`: Time series analysis (ADF test, ARIMA).
  - `scipy.stats`: Statistical tests (Shapiro-Wilk, skewness, kurtosis).
  - `arch`: GARCH model fitting.
- **Plotly**: Interactive charts for better user experience.

---

## **Installation**

Follow these steps to run the application locally:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/stock-price-analysis.git
   cd stock-price-analysis


# Usage
  - Enter the stock ticker symbol (e.g., AAPL for Apple, MSFT for Microsoft).
  - Specify the start date and end date for the analysis.
  - Explore:
    Distribution and normality of stock prices.
    ACF and PACF plots to understand autocorrelations.
    Stationarity using the ADF test.
  - View:
    ARIMA model forecasts and historical trends.
    Rolling volatility trends and GARCH-based volatility forecasts.
    Calculate financial metrics like mean returns, skewness, and kurtosis.



# Dashboard Overview

  ![image](https://github.com/user-attachments/assets/9c6195b3-456e-40d4-b599-b3d67618289e)



# Stock Distribution Analysis

![image](https://github.com/user-attachments/assets/5e381cee-9713-4974-8db3-11a8f388a39d)


# Time Series Forecasting

![image](https://github.com/user-attachments/assets/6e2bb134-f1ee-4553-8178-d0a0c769b97f)


License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments
Streamlit for building interactive dashboards.
Yahoo Finance for stock data retrieval.
Statsmodels for statistical modeling.
ARCH package for volatility modeling.

