# Stock Price Prediction Using Advanced Machine Learning Techniques

## Project Overview
The goal of this project is to develop an advanced machine learning model to predict stock price movements based on historical stock prices, trading volume, and external financial indicators. The project aims to outperform traditional "buy-and-hold" strategies by leveraging deep learning, time-series analysis, and sentiment analysis from financial news.

## Project Goals
- Successfully predict next-day stock price movement.
- Compare different modeling approaches (LSTM, Transformer, XGBoost, GARCH, and hybrid models).
- Develop an **interactive visualization dashboard** to display stock trends and sentiment analysis.
- Provide a **reproducible pipeline** that allows researchers to analyze financial data efficiently.

## Data Collection
We will collect data from the following sources:

1. **Stock Price Data**
   - **Source**: Yahoo Finance API
   - **Collection Method**: API calls to retrieve daily historical stock prices, adjusted close prices, and volume data.

2. **Trading Volume & Market Indicators**
   - **Source**: Alpha Vantage and Quandl
   - **Collected Data**: VIX (Volatility Index), S&P 500 index movements, bond yields.
   - **Collection Method**: Scheduled API requests with caching to avoid rate limits.

3. **Sentiment Analysis Data**
   - **Sources**: Bloomberg, Reuters (web scraping), Twitter API
   - **Collected Data**: Financial news headlines and social media sentiment related to selected stocks.
   - **Collection Method**:
     - Web scraping with BeautifulSoup or Scrapy for financial news.
     - Twitter API for real-time sentiment analysis (if API access is granted).
   - **Preprocessing Considerations**: Handling API rate limits and removing duplicate news articles.

## Data Cleaning
- **Handling missing values**: Forward filling, interpolation, or mean imputation.
- **Outlier detection**: Removing extreme price fluctuations caused by stock splits or anomalies.
- **Normalization**: Standardizing numerical data to improve model performance.

## Feature Extraction
We will generate key features to enhance model learning:
- **Technical Indicators**: Moving Averages, RSI, MACD, Bollinger Bands.
- **Sentiment Features**: Extracted from financial headlines using NLP models (BERT, VADER, ChatGPT sentiment API).
- **Time-based Features**: Weekday, holiday effects, earnings release impact.

## Data Visualization
We will create multiple visualizations to explore stock trends and model insights:
- **Interactive K-line (candlestick) charts** to track stock price movements.
- **Correlation heatmaps** to analyze feature relationships.
- **Dual-axis line charts** to compare sentiment analysis trends with stock price changes.
- **Model performance plots** comparing accuracy (MSE, R², Sharpe Ratio) across different models.
- **Interactive dashboards (Plotly, Dash)** allowing users to dynamically select stocks and timeframes.

## Model Training
The following models will be implemented and compared:
1. **Baseline Model**: Simple Moving Average for naive prediction.
2. **Traditional Machine Learning Models**: XGBoost, Random Forest.
3. **Deep Learning Approaches**:
   - **LSTM (Long Short-Term Memory Networks)**: Capturing sequential dependencies in stock prices.
   - **Transformer-based Models**: Leveraging attention mechanisms for forecasting.
   - **Hybrid Models**: Combining CNN and LSTM for feature extraction and time-series prediction.

## Test Plan
- **Data Splitting**: 80% training, 20% test set.
- **Cross-validation**: Rolling window validation to ensure performance stability.
- **Evaluation Metrics**:
  - **Mean Squared Error (MSE)** – Measures prediction accuracy.
  - **R-squared (R²)** – Assesses model fit.
  - **Sharpe Ratio** – Evaluates financial performance.

## Reproducibility and GitHub Workflow
- The project will be maintained on GitHub with the following structure:
- README.md: Project details, setup, and usage instructions.
- Notebooks/: Jupyter notebooks for data preprocessing and visualization.
- src/: Python scripts for data collection, model training, and evaluation.
- tests/: Unit tests for data processing and model evaluation.
- **CI/CD with GitHub Actions**:
- Automated testing for data processing and model evaluation.
- Linting and code formatting with pre-commit hooks.

## Timeline
| Date      | Task                                   |
|-----------|--------------------------------------|
| **Feb 10**  | Proposal submission                   |
| **Mar 31**  | Midterm report with preliminary results |
| **May 1**   | Final report and presentation        |

## Expected Outcome
- A machine learning pipeline capable of predicting stock price movements.
- Comparison of various prediction models with performance benchmarks.
- Interactive visualizations to interpret stock market behavior.
- A reproducible GitHub repository with clear documentation.

## Submission Details
- **GitHub Repository**: [Project Repository Link]
- **Midterm & Final Presentation**: Recorded and uploaded to YouTube.
- **README.md**: Comprehensive documentation including project setup, instructions, and results.


Reference:
https://people.duke.edu/~rnau/411arim.htm
https://documentation.sas.com/doc/en/capcdc/v_025/vfcdc/vfug/p141sb29rauf5pn1d2j7v5l9mfcs.htm
https://www.math.pku.edu.cn/teachers/lidf/course/fts/ftsnotes/html/_ftsnotes/fts-arma.html#arma-concept