import os
import shutil
import sys
import time
import subprocess

def ensure_dir(dir_path):
    """Create directory if it doesn't exist"""
    os.makedirs(dir_path, exist_ok=True)

def safe_remove_directory(directory):
    """Try to safely remove a directory, even if it's locked by Windows"""
    if not os.path.exists(directory):
        return True
    
    print(f"Attempting to remove {directory}...")
    
    # First try with shutil.rmtree
    try:
        shutil.rmtree(directory)
        return True
    except (PermissionError, OSError) as e:
        print(f"Standard removal failed: {e}")
    
    # If that fails, try with Windows commands
    try:
        print("Trying with Windows commands...")
        # Try using rd /s /q which is sometimes more effective with locked files
        subprocess.run(['rd', '/s', '/q', directory], shell=True, check=False)
        time.sleep(1)  # Give Windows a moment
        
        if not os.path.exists(directory):
            return True
        
        # If that doesn't work, try creating a new directory with different name
        if os.path.exists(directory):
            print(f"Creating a new directory instead...")
            return False
        
        return True
    except Exception as e:
        print(f"Windows command removal failed: {e}")
        return False

def copy_files(src_dir, dst_dir, exclude_dirs=None, exclude_files=None):
    """Copy files and directories, excluding specific ones"""
    if exclude_dirs is None:
        exclude_dirs = ['.venv', 'env', 'venv', 'ENV', '__pycache__', '.git', '.idea', '.vscode']
    
    if exclude_files is None:
        exclude_files = []
    
    print(f"Copying from {src_dir} to {dst_dir}...")
    
    # Create destination if it doesn't exist
    ensure_dir(dst_dir)
    
    # List all items in source directory
    for item in os.listdir(src_dir):
        src_path = os.path.join(src_dir, item)
        dst_path = os.path.join(dst_dir, item)
        
        # Skip the CS506-Final-Project directory we're creating
        if os.path.isdir(src_path) and item == "CS506-Final-Project":
            print(f"Skipping our target directory: {src_path}")
            continue
        
        # Skip excluded directories
        if os.path.isdir(src_path) and item in exclude_dirs:
            print(f"Skipping excluded directory: {src_path}")
            continue
        
        # Skip excluded files
        if os.path.isfile(src_path) and item in exclude_files:
            print(f"Skipping excluded file: {src_path}")
            continue
        
        # Skip very large files (>100MB)
        if os.path.isfile(src_path) and os.path.getsize(src_path) > 100 * 1024 * 1024:
            print(f"Skipping large file: {src_path}")
            continue
        
        # If directory, recursively copy
        if os.path.isdir(src_path):
            copy_files(src_path, dst_path, exclude_dirs, exclude_files)
        else:
            # Copy the file
            shutil.copy2(src_path, dst_path)
            print(f"Copied: {item}")

def create_repo_structure():
    """Create repository structure for GitHub"""
    # Base repository directory
    repo_dir = "CS506-Final-Project"
    
    # Clean up existing directory if it exists
    if os.path.exists(repo_dir):
        success = safe_remove_directory(repo_dir)
        if not success:
            # If can't remove, use an alternative name
            timestamp = time.strftime("%Y%m%d%H%M%S")
            repo_dir = f"CS506-Final-Project-{timestamp}"
            print(f"Using alternative directory name: {repo_dir}")
    
    # Create repository structure
    ensure_dir(repo_dir)
    ensure_dir(os.path.join(repo_dir, "lstm"))
    ensure_dir(os.path.join(repo_dir, "data_raw"))
    ensure_dir(os.path.join(repo_dir, "data_processed"))
    ensure_dir(os.path.join(repo_dir, "data_preprocess"))
    
    # Copy everything to lstm directory
    copy_files(".", os.path.join(repo_dir, "lstm"))
    
    # Create .gitignore file
    gitignore_content = """
# Virtual environments
.venv/
venv/
env/
ENV/

# Python cache
__pycache__/
*.py[cod]
*$py.class

# IDE files
.idea/
.vscode/

# OS specific
.DS_Store
Thumbs.db

# Don't exclude any output files - we want to include all results
"""
    
    with open(os.path.join(repo_dir, ".gitignore"), "w") as f:
        f.write(gitignore_content)
    
    # Create main README.md
    readme_content = """# CS506-Final-Project: Stock Price Prediction

## Project Overview

This repository contains the implementation of a stock price prediction system using machine learning techniques. The primary focus is on using LSTM (Long Short-Term Memory) neural networks for time-series forecasting of stock prices.

## Repository Structure

- `/lstm`: LSTM-based stock price prediction system
  - Implementation of neural network models for stock prediction
  - Ensemble methods for improving prediction accuracy
  - Visualization tools for analyzing results
  - Output files demonstrating model performance
  
- `/data_preprocess`: Code for preprocessing raw stock data
  
- `/data_processed`: Processed datasets ready for model training
  
- `/data_raw`: Sample raw data files

For detailed information about the LSTM implementation, see the README.md file in the `/lstm` directory.
"""
    
    with open(os.path.join(repo_dir, "README.md"), "w") as f:
        f.write(readme_content)
    
    # Add .gitkeep files to empty directories
    for empty_dir in ["data_raw", "data_processed", "data_preprocess"]:
        with open(os.path.join(repo_dir, empty_dir, ".gitkeep"), "w") as f:
            pass
    
    print("\nRepository structure created successfully!")
    print(f"\nFiles copied to {repo_dir}/lstm/")
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
    create_repo_structure() 