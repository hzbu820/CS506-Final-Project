import sys
import os

# Add the parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Try importing from the renamed module
try:
    from scripts.visualize_intraday_data import download_intraday_data, calculate_technical_indicators
    print("Import successful!")
except ImportError as e:
    print(f"Import error: {e}")

print("Test complete") 