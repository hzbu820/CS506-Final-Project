#!/usr/bin/env python3
"""
System check utility for the stock analysis tool.
This script verifies system requirements and configuration.
"""

import sys
import os
import platform
import datetime
import requests
import subprocess
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 6):
        logger.error(f"Python version {version.major}.{version.minor} is not supported. Please use Python 3.6+")
        return False
    logger.info(f"Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def check_system_date():
    """Check if system date is reasonable."""
    now = datetime.datetime.now()
    current_year = now.year
    
    # Check if year is reasonable (between 2023 and 2025)
    if current_year < 2023 or current_year > 2025:
        logger.error(f"System date appears incorrect: {now.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.error(f"Year {current_year} is outside the expected range (2023-2025)")
        
        # Try to get actual date from internet
        try:
            response = requests.get("http://worldtimeapi.org/api/ip")
            if response.status_code == 200:
                internet_time = response.json().get('datetime', '')
                logger.info(f"Internet time reports: {internet_time}")
                logger.error("Please correct your system date and time")
            else:
                logger.error("Could not verify correct time from internet")
        except Exception as e:
            logger.error(f"Error checking internet time: {str(e)}")
        
        return False
    
    logger.info(f"System date: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    return True

def check_internet_connection():
    """Check if there's an active internet connection."""
    try:
        response = requests.get("https://www.google.com", timeout=5)
        if response.status_code == 200:
            logger.info("Internet connection: OK")
            return True
        else:
            logger.error(f"Internet connection check failed with status code: {response.status_code}")
            return False
    except requests.RequestException as e:
        logger.error(f"Internet connection check failed: {str(e)}")
        return False

def check_dependencies():
    """Check if required packages are installed."""
    required_packages = ["yfinance", "pandas", "requests", "json5"]
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"Package {package}: OK")
        except ImportError:
            logger.error(f"Package {package}: Not found")
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing packages: {', '.join(missing_packages)}")
        logger.info("Install missing packages with: pip install -r requirements.txt")
        return False
    
    return True

def main():
    """Run all system checks."""
    logger.info("Running system checks...")
    
    checks = [
        ("Python version", check_python_version()),
        ("System date", check_system_date()),
        ("Internet connection", check_internet_connection()),
        ("Dependencies", check_dependencies())
    ]
    
    # Print summary
    logger.info("\nSystem Check Summary:")
    all_passed = True
    for name, result in checks:
        status = "PASS" if result else "FAIL"
        logger.info(f"{name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        logger.info("\nAll checks passed! The system is ready to run the stock analysis tool.")
        return 0
    else:
        logger.error("\nSome checks failed. Please fix the issues before running the stock analysis tool.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 