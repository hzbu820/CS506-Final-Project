#!/usr/bin/env python
"""
GitHub Push Script - Pushes the LSTM Stock Prediction project to GitHub
Excludes virtual environment and other unnecessary files
"""

import os
import subprocess
import sys
import argparse
from datetime import datetime


def run_command(command, cwd=None):
    """Run a shell command and print the output"""
    print(f"Running: {command}")
    process = subprocess.Popen(
        command, 
        shell=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd
    )
    stdout, stderr = process.communicate()
    
    if stdout:
        print(stdout)
    if stderr:
        print(f"Error: {stderr}")
    
    return process.returncode == 0


def create_gitignore():
    """Create or update .gitignore file with appropriate exclusions"""
    gitignore_content = """# Virtual Environment
.venv/
venv/
env/
ENV/

# Python cache files
__pycache__/
*.py[cod]
*$py.class
.pytest_cache/
.coverage
htmlcov/

# Distribution / packaging
dist/
build/
*.egg-info/

# IDE files
.idea/
.vscode/
*.swp
*.swo

# OS specific files
.DS_Store
Thumbs.db

# Large data files and model files
*.h5
*.pkl
*.weights
large_data_files/

# Log files
logs/
*.log

# Jupyter Notebook
.ipynb_checkpoints

# Environment variables
.env
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)
    
    print("Created .gitignore file")


def setup_git_repository(repo_url, branch_name="main"):
    """Initialize git repository and set remote"""
    # Check if .git directory exists
    if not os.path.exists('.git'):
        if not run_command("git init"):
            print("Failed to initialize git repository")
            return False
    
    # Create .gitignore
    create_gitignore()
    
    # Set remote repository URL if provided
    if repo_url:
        if not run_command(f'git remote remove origin 2>/dev/null || true'):
            print("Warning: Could not remove existing origin (this may be normal)")
        
        if not run_command(f'git remote add origin {repo_url}'):
            print(f"Failed to add remote: {repo_url}")
            return False
    
    return True


def commit_and_push(commit_message=None, branch_name="main"):
    """Commit all changes and push to remote"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if commit_message is None:
        commit_message = f"LSTM Stock Prediction project update - {timestamp}"
    
    # Add all files
    if not run_command("git add ."):
        print("Failed to add files to repository")
        return False
    
    # Commit changes
    if not run_command(f'git commit -m "{commit_message}"'):
        print("Nothing to commit or commit failed")
        # Continue anyway as there might just be no changes
    
    # Create branch if it doesn't exist
    run_command(f"git checkout -b {branch_name} 2>/dev/null || git checkout {branch_name}")
    
    # Push to remote
    push_command = f"git push -u origin {branch_name}"
    if not run_command(push_command):
        print(f"Failed to push to remote. Attempting force push...")
        # Try force push as a fallback
        if not run_command(f"git push -u origin {branch_name} --force"):
            print("Force push also failed. Please check your repository settings and credentials.")
            return False
    
    return True


def main():
    """Main function to push to GitHub"""
    parser = argparse.ArgumentParser(description="Push LSTM Stock Prediction project to GitHub")
    parser.add_argument("--repo", help="GitHub repository URL (e.g., https://github.com/username/repo.git)")
    parser.add_argument("--branch", default="main", help="Branch name to push to (default: main)")
    parser.add_argument("--message", help="Custom commit message")
    args = parser.parse_args()
    
    # Welcome message
    print("=" * 70)
    print("LSTM Stock Prediction Project - GitHub Push Script")
    print("=" * 70)
    
    # Get repository URL from args or prompt user
    repo_url = args.repo
    if not repo_url:
        repo_url = input("Enter GitHub repository URL (e.g., https://github.com/username/repo.git): ")
    
    # Get branch name
    branch_name = args.branch
    
    # Get commit message
    commit_message = args.message
    
    # Setup git repository
    print("\nSetting up Git repository...")
    if not setup_git_repository(repo_url, branch_name):
        print("Failed to set up Git repository. Exiting.")
        return 1
    
    # Commit and push changes
    print("\nCommitting and pushing changes...")
    if commit_and_push(commit_message, branch_name):
        print("\nSuccess! Your LSTM Stock Prediction project has been pushed to GitHub.")
        print(f"Repository URL: {repo_url}")
        print(f"Branch: {branch_name}")
    else:
        print("\nFailed to push the project to GitHub.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 