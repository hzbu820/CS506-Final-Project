import os
import shutil
import subprocess
import sys
import tempfile

def run_command(command, cwd=None):
    """Run a shell command and return output"""
    print(f"Running: {command}")
    process = subprocess.run(command, shell=True, cwd=cwd, 
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if process.returncode != 0:
        print(f"Error executing command: {command}")
        print(f"Output: {process.stdout}")
        print(f"Error: {process.stderr}")
        return False
    return True

def create_gitignore(repo_path):
    """Create a .gitignore file to exclude virtual environments and large files"""
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

# Don't exclude any output files
# We want to keep all figures and results
"""
    with open(os.path.join(repo_path, '.gitignore'), 'w') as f:
        f.write(gitignore_content)

def copy_project(src_dir, dst_dir, exclude_dirs=None):
    """Copy project files excluding specific directories"""
    if exclude_dirs is None:
        exclude_dirs = ['.venv', '__pycache__', '.git', '.idea', '.vscode']
    
    for item in os.listdir(src_dir):
        if item in exclude_dirs:
            continue
        
        s = os.path.join(src_dir, item)
        d = os.path.join(dst_dir, item)
        
        if os.path.isdir(s):
            if not os.path.exists(d):
                os.makedirs(d)
            copy_project(s, d, exclude_dirs)
        else:
            # Skip very large files
            if os.path.getsize(s) > 100 * 1024 * 1024:  # 100MB
                print(f"Skipping large file: {s}")
                continue
            shutil.copy2(s, d)

def main():
    """Main function to push to GitHub"""
    # Get current directory
    current_dir = os.getcwd()
    print(f"Current directory: {current_dir}")
    
    # Create temporary directory for GitHub repo
    temp_dir = tempfile.mkdtemp()
    print(f"Created temporary directory: {temp_dir}")
    
    # Clone repository
    github_url = "https://github.com/hzbu820/CS506-Final-Project.git"
    clone_cmd = f"git clone {github_url} {temp_dir}"
    if not run_command(clone_cmd):
        print("Failed to clone repository")
        return
    
    # Switch to the feature branch or create if doesn't exist
    branch_name = "my-feature-branch"
    checkout_cmd = f"git checkout {branch_name} || git checkout -b {branch_name}"
    if not run_command(checkout_cmd, cwd=temp_dir):
        print(f"Failed to checkout branch {branch_name}")
        return
    
    # Create lstm directory
    lstm_dir = os.path.join(temp_dir, "lstm")
    if not os.path.exists(lstm_dir):
        os.makedirs(lstm_dir)
    
    # Copy project files to lstm directory
    print(f"Copying project files to {lstm_dir}...")
    copy_project(current_dir, lstm_dir)
    
    # Create .gitignore
    create_gitignore(temp_dir)
    
    # Setup Git config
    run_command("git config user.email \"placeholder@example.com\"", cwd=temp_dir)
    run_command("git config user.name \"LSTM Project\"", cwd=temp_dir)
    
    # Add files to Git
    run_command("git add .", cwd=temp_dir)
    
    # Commit changes
    commit_cmd = "git commit -m \"Add LSTM Stock Price Prediction System with outputs\""
    if not run_command(commit_cmd, cwd=temp_dir):
        print("No changes to commit or commit failed")
    
    # Push to GitHub (force push to overwrite)
    push_cmd = f"git push -f origin {branch_name}"
    if not run_command(push_cmd, cwd=temp_dir):
        print("Failed to push to GitHub")
        return
    
    print("Successfully pushed to GitHub!")
    print(f"Check your branch at: {github_url}/tree/{branch_name}")
    
    # Clean up temporary directory
    try:
        shutil.rmtree(temp_dir)
        print(f"Cleaned up temporary directory: {temp_dir}")
    except:
        print(f"Failed to clean up temporary directory: {temp_dir}")

if __name__ == "__main__":
    main() 