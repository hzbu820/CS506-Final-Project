import yfinance as yf

df = yf.download("AAPL", start="2023-03-01", end="2024-03-01", interval="1h")
print(df.head())
