import yfinance as yf
import json
import time
import logging
import sys
import os
import requests
from datetime import datetime, timedelta
from config import STOCK_TICKERS, START_DATE, END_DATE, INTERVAL, MAX_RETRIES, REQUEST_TIMEOUT

# Define output directory
OUTPUT_DIR = "output"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def check_internet_connection():
    """Check if there's an active internet connection by pinging a reliable site."""
    try:
        # Try to connect to Google's DNS server
        requests.get('https://8.8.8.8', timeout=3)
        return True
    except requests.ConnectionError:
        logger.error("No internet connection available.")
        return False
    except Exception as e:
        logger.error(f"Error checking internet connection: {str(e)}")
        return False

def validate_dates(start_date, end_date):
    """
    Validate and adjust date range if necessary.
    
    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        
    Returns:
        tuple: Validated (start_date, end_date)
    """
    # Check system date first
    system_year = datetime.now().year
    if system_year > 2024:
        logger.warning(f"System year appears to be set to {system_year}, which may be incorrect.")
        logger.warning("This could cause issues with date validation and API requests.")
    
    today = datetime.now().strftime("%Y-%m-%d")
    
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        current_dt = datetime.now()
        
        # Check if end date is in the future
        if end_dt > current_dt:
            logger.warning(f"End date {end_date} is in the future. Using current date instead.")
            end_date = today
            end_dt = current_dt
            
        # Check if start date is after end date
        if start_dt > end_dt:
            logger.warning(f"Start date {start_date} is after end date {end_date}. Swapping dates.")
            start_date, end_date = end_date, start_date
            
        # Check if date range is too short
        if (end_dt - start_dt).days < 2:
            logger.warning("Date range too short. Setting range to last 30 days.")
            end_date = today
            start_date = (current_dt - timedelta(days=30)).strftime("%Y-%m-%d")
            
    except ValueError as e:
        logger.error(f"Invalid date format: {str(e)}. Using last 30 days.")
        end_date = today
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        
    return start_date, end_date

def fetch_multiple_intervals(ticker_symbol):
    """
    Fetch data for multiple intervals for a single ticker.
    Returns data frames containing OHLCV data for different timeframes.
    """
    logger.info(f"Fetching multiple interval data for {ticker_symbol}...")
    
    # Define intervals and periods to fetch
    intervals = [
        {"interval": "1m", "period": "7d", "description": "1-minute data for last 7 days"},
        {"interval": "5m", "period": "60d", "description": "5-minute data for last 60 days"},
        {"interval": "15m", "period": "60d", "description": "15-minute data for last 60 days"},
        {"interval": "30m", "period": "60d", "description": "30-minute data for last 60 days"},
        {"interval": "60m", "period": "730d", "description": "60-minute data for last 2 years"},
        {"interval": "1d", "period": "max", "description": "Daily data for maximum period"}
    ]
    
    results = {}
    ticker = yf.Ticker(ticker_symbol)
    
    for interval_config in intervals:
        interval = interval_config["interval"]
        period = interval_config["period"]
        description = interval_config["description"]
        
        try:
            logger.info(f"Fetching {description} for {ticker_symbol}")
            df = ticker.history(interval=interval, period=period)
            
            if df.empty:
                logger.warning(f"No data returned for {ticker_symbol} with interval {interval}")
                results[interval] = None
            else:
                logger.info(f"Successfully fetched {len(df)} rows of {interval} data for {ticker_symbol}")
                results[interval] = df
                
        except Exception as e:
            logger.error(f"Error fetching {interval} data for {ticker_symbol}: {str(e)}")
            results[interval] = None
            
    return results

def save_interval_data(ticker, interval, data, base_dir="output"):
    """
    Save data for a specific ticker and interval to a separate JSON file.
    
    Args:
        ticker (str): Stock ticker symbol
        interval (str): Time interval
        data (dict): Data to save
        base_dir (str): Base directory for output files
    """
    # Create directory structure: output/TICKER/
    ticker_dir = os.path.join(base_dir, ticker)
    os.makedirs(ticker_dir, exist_ok=True)
    
    # Create filename: TICKER_INTERVAL_YYYYMMDD.json
    timestamp = datetime.now().strftime('%Y%m%d')
    filename = f"{ticker}_{interval}_{timestamp}.json"
    filepath = os.path.join(ticker_dir, filename)
    
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved {interval} data for {ticker} to {filepath}")
    except Exception as e:
        logger.error(f"Error saving {interval} data for {ticker}: {str(e)}")

def analyze_data(data_dict):
    """
    Analyze stock data across multiple intervals.
    
    Args:
        data_dict (dict): Dictionary of DataFrames for different intervals
        
    Returns:
        dict: Dictionary containing analysis results for each interval
    """
    analysis = {}
    
    for interval, df in data_dict.items():
        if df is not None and not df.empty:
            try:
                interval_analysis = {
                    'start_date': df.index[0].strftime('%Y-%m-%d %H:%M:%S'),
                    'end_date': df.index[-1].strftime('%Y-%m-%d %H:%M:%S'),
                    'data_points': len(df),
                    'price_data': {
                        'open': df['Open'].tolist(),
                        'high': df['High'].tolist(),
                        'low': df['Low'].tolist(),
                        'close': df['Close'].tolist(),
                        'volume': df['Volume'].tolist()
                    },
                    'timestamps': [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in df.index],
                    'summary': {
                        'avg_close': float(df['Close'].mean()),
                        'max_high': float(df['High'].max()),
                        'min_low': float(df['Low'].min()),
                        'volume_total': int(df['Volume'].sum()),
                        'volume_avg': float(df['Volume'].mean()),
                        'price_volatility': float(df['Close'].std())
                    }
                }
                
                # Calculate price change and percentage
                if len(df) > 1:
                    first_close = df['Close'].iloc[0]
                    last_close = df['Close'].iloc[-1]
                    price_change = last_close - first_close
                    price_change_pct = (price_change / first_close) * 100
                    interval_analysis['summary']['price_change'] = float(price_change)
                    interval_analysis['summary']['price_change_pct'] = float(price_change_pct)
                
                analysis[interval] = interval_analysis
                
            except Exception as e:
                logger.error(f"Error analyzing {interval} data: {str(e)}")
                analysis[interval] = {'error': str(e)}
        else:
            analysis[interval] = {'error': 'No data available'}
            
    return analysis

def fetch_stock_data(ticker, start_date, end_date, interval=INTERVAL, retries=MAX_RETRIES):
    """
    Fetch stock data for a given ticker from Yahoo Finance.
    Now returns data for multiple intervals instead of just one.
    """
    if not check_internet_connection():
        logger.error("Cannot fetch stock data without internet connection.")
        return None
    
    # Validate dates before fetching
    start_date, end_date = validate_dates(start_date, end_date)
    
    # Fetch data for multiple intervals
    return fetch_multiple_intervals(ticker)

def main():
    """Main function to run the financial data analysis."""
    # Validate and potentially adjust date range
    validated_start, validated_end = validate_dates(START_DATE, END_DATE)
    
    # Create summary results
    summary_results = {
        'analysis_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'period': {
            'start': validated_start,
            'end': validated_end
        },
        'stocks': {}
    }
    
    for ticker in STOCK_TICKERS:
        logger.info(f"Processing {ticker}...")
        data = fetch_stock_data(ticker, validated_start, validated_end)
        
        if data is not None:
            logger.info(f"Analyzing data for {ticker}...")
            analysis = analyze_data(data)
            
            # Save each interval to a separate file
            for interval, interval_data in analysis.items():
                if 'error' not in interval_data:
                    save_interval_data(ticker, interval, interval_data)
            
            # Store summary in main results
            summary_results['stocks'][ticker] = {
                interval: {
                    'summary': data.get('summary', {}),
                    'error': data.get('error', None)
                } for interval, data in analysis.items()
            }
        else:
            logger.warning(f"No data available for {ticker} in the specified date range.")
            summary_results['stocks'][ticker] = {'error': 'No data available for the specified date range'}
    
    # Save summary results
    try:
        summary_file = os.path.join(OUTPUT_DIR, f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(summary_file, "w") as f:
            json.dump(summary_results, f, indent=2)
        
        logger.info(f"Analysis complete. Summary saved to {summary_file}")
        
    except Exception as e:
        logger.error(f"Error saving summary results: {str(e)}")
        
    return summary_results

if __name__ == "__main__":
    main()