"""
Script to check and fix dependency loading issues.
Run this script to verify that all necessary dependencies can be imported correctly.
"""

import os
import sys
import time
import importlib
import subprocess

def check_import(module_name, description):
    """
    Check if a module can be imported and provide feedback
    """
    print(f"Checking {module_name} ({description})...", end="")
    sys.stdout.flush()
    
    try:
        importlib.import_module(module_name)
        print(" ✓")
        return True
    except ImportError as e:
        print(f" ✗ - {str(e)}")
        return False

def install_package(package_name):
    """
    Install a package using pip
    """
    print(f"Installing {package_name}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"Successfully installed {package_name}")
        return True
    except subprocess.CalledProcessError:
        print(f"Failed to install {package_name}")
        return False

def main():
    """
    Main function to check and fix dependencies
    """
    print("=" * 60)
    print("Dependency Check and Fix Utility")
    print("=" * 60)
    print("This script will check if all required packages can be imported.")
    print("If any issues are found, it will attempt to resolve them.")
    
    dependencies = [
        ("numpy", "Core numerical computing library"),
        ("pandas", "Data manipulation library"),
        ("matplotlib", "Plotting library"),
        ("torch", "PyTorch deep learning framework"),
        ("sklearn", "Machine learning library"),
        ("scipy", "Scientific computing library"),
        ("yfinance", "Yahoo Finance data retrieval"),
        ("statsmodels", "Statistical models"),
    ]
    
    # Check each dependency
    all_good = True
    failed_imports = []
    
    for package, description in dependencies:
        if not check_import(package, description):
            all_good = False
            failed_imports.append(package)
    
    # If any imports failed, offer to install them
    if not all_good:
        print("\nSome dependencies could not be imported.")
        
        fix_option = input("\nWould you like to try installing the missing packages? (y/n): ")
        
        if fix_option.lower() == 'y':
            for package in failed_imports:
                install_package(package)
                
            # Check if installations were successful
            print("\nRe-checking imports after installation:")
            still_failed = []
            for package in failed_imports:
                if not check_import(package, ""):
                    still_failed.append(package)
            
            if still_failed:
                print("\nThe following packages still have import issues:")
                for package in still_failed:
                    print(f"  - {package}")
                print("\nTry installing these packages manually or check for conflicts.")
            else:
                print("\nAll dependencies are now available!")
        else:
            print("\nNo packages were installed.")
    else:
        print("\nAll dependencies are available!")
    
    # Check import strategies
    print("\nTesting scipy modules with alternate import methods...")
    
    # Test individual imports to identify problematic modules
    scipy_modules = [
        "scipy.stats", 
        "scipy.optimize", 
        "scipy.interpolate",
        "scipy.linalg"
    ]
    
    for module in scipy_modules:
        time.sleep(0.5)  # Short delay to avoid import rush
        print(f"Testing import of {module}...", end="")
        sys.stdout.flush()
        
        try:
            importlib.import_module(module)
            print(" ✓")
        except Exception as e:
            print(f" ✗ - {str(e)}")
    
    print("\nSuggested modifications to your scripts:")
    print("1. Add 'import time' at the top of scripts")
    print("2. Add short sleep delays between heavy library imports: time.sleep(0.5)")
    print("3. Import only needed functions from scipy instead of entire modules")
    print("4. Consider adding a try/except block for imports that may fail")
    
    print("\nExample:")
    print("""
    import numpy as np
    import pandas as pd
    import time
    
    # Add delay between heavy imports
    time.sleep(0.5)
    
    # Import specific functions instead of entire modules
    try:
        from scipy.stats import linregress
        from scipy.optimize import minimize
    except ImportError:
        print("Warning: Some scipy functions could not be imported")
    """)
    
    print("\nTest complete!")

if __name__ == "__main__":
    main() 