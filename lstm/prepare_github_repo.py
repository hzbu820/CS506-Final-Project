import os
import shutil
import sys
import glob

def ensure_dir(dir_path):
    """Create directory if it doesn't exist"""
    os.makedirs(dir_path, exist_ok=True)

def copy_with_structure(src_dir, dst_dir, exclude_dirs=None):
    """Copy directory structure and files, excluding specific directories"""
    if exclude_dirs is None:
        exclude_dirs = ['.git', '.venv', '__pycache__', '.idea', '.vscode']
    
    # Create the destination directory if it doesn't exist
    ensure_dir(dst_dir)
    
    # Walk through source directory
    for root, dirs, files in os.walk(src_dir):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        # Calculate the relative path
        rel_path = os.path.relpath(root, src_dir)
        if rel_path == '.':
            rel_path = ''
        
        # Create the corresponding directory in the destination
        dst_path = os.path.join(dst_dir, rel_path)
        ensure_dir(dst_path)
        
        # Copy files
        for file in files:
            src_file = os.path.join(root, file)
            dst_file = os.path.join(dst_path, file)
            
            # Skip large files
            if os.path.getsize(src_file) > 100 * 1024 * 1024:  # Skip files larger than 100MB
                print(f"Skipping large file: {src_file}")
                continue
                
            shutil.copy2(src_file, dst_file)

def main():
    """Copy project to CS506-Final-Project structure"""
    print("Preparing LSTM project with outputs for GitHub...")
    
    # Base directory for GitHub repository
    repo_dir = 'CS506-Final-Project'
    
    # Create the repository directory
    ensure_dir(repo_dir)
    
    # Create the lstm directory in the repository
    lstm_dir = os.path.join(repo_dir, 'lstm')
    ensure_dir(lstm_dir)
    
    # Copy everything except excluded directories
    exclude_dirs = [
        '.git', '.venv', '__pycache__', '.idea', '.vscode', 
        'CS506-Final-Project'  # Don't copy into itself
    ]
    
    # Copy the entire project to the lstm directory
    copy_with_structure('.', lstm_dir, exclude_dirs)
    
    # Create empty directories for the other parts of the repository
    dirs_to_create = ['data_raw', 'data_processed', 'data_preprocess']
    for dir_name in dirs_to_create:
        ensure_dir(os.path.join(repo_dir, dir_name))
        # Create a .gitkeep file
        with open(os.path.join(repo_dir, dir_name, '.gitkeep'), 'w') as f:
            pass
    
    # Create main README.md
    readme_content = """# CS506-Final-Project: Stock Price Prediction

## Project Overview

This repository contains the implementation of a stock price prediction system using machine learning techniques. The primary focus is on using LSTM (Long Short-Term Memory) neural networks for time-series forecasting of stock prices.

## Repository Structure

- `/lstm`: LSTM-based stock price prediction system
  - Implementation of neural network models for stock prediction
  - Ensemble methods for improving prediction accuracy
  - Visualization tools for analyzing results
  - Output files from model training and testing
  
- `/data_preprocess`: Code for preprocessing raw stock data
  
- `/data_processed`: Processed datasets ready for model training
  
- `/data_raw`: Sample raw data files

For detailed information about the LSTM implementation, see the README.md file in the `/lstm` directory.
"""
    
    with open(os.path.join(repo_dir, 'README.md'), 'w') as f:
        f.write(readme_content)
    
    print("\nProject prepared for GitHub with outputs included!")
    print(f"Files are copied to {repo_dir}/lstm/")
    print("\nNext steps:")
    print(f"1. cd {repo_dir}")
    print("2. git init")
    print("3. git checkout -b my-feature-branch")
    print("4. git add .")
    print("5. git commit -m \"Add LSTM Stock Price Prediction System with outputs\"")
    print("6. git remote add origin https://github.com/hzbu820/CS506-Final-Project.git")
    print("7. git fetch origin")
    print("8. git push --force origin my-feature-branch")

if __name__ == "__main__":
    main() 