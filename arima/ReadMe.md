ARIMA (AutoRegressive Integrated Moving Average) Model
The ARIMA(p, d, q) model is a widely used statistical method for time series forecasting. It consists of three key parameters:

p (AutoRegressive term): The number of lag observations included in the model (i.e., how many past data points are used to predict the current value).

d (Differencing order): The number of times the raw observations are differenced to make the time series stationary.

If the data is already stationary, we set d = 0.

Otherwise, we apply differencing to remove trends and seasonality.

q (Moving Average term): The size of the moving average window, representing the number of lagged forecast errors (white noise) used to improve predictions.

Why We Chose ARIMA?
We selected ARIMA as a baseline model for our project due to its simplicity and interpretability. It serves as a solid starting point to model univariate time series data, especially when the data exhibits short-term dependencies.

Our preprocessing involves:

Testing for stationarity.

Applying differencing when necessary to achieve stationarity.

Fitting the ARIMA model based on optimal (p, d, q) values.

Limitations of ARIMA
While ARIMA is a strong classical method, it has several limitations:

1. Short-range memory: ARIMA is not designed to capture long-term dependencies or complex temporal patterns.

2. Univariate only: It cannot model multivariate relationships without extensions.

3. Assumes linearity: ARIMA works best when the underlying data generation process is linear.

Because of these constraints, we later explore more advanced models such as LSTM that better capture non-linearity and long-term trends.


