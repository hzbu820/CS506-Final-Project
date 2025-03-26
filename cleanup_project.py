import os
import shutil
import sys

def cleanup_directory(directory, exclude_dirs=None, exclude_files=None):
    """Remove redundant and temporary files from a directory"""
    if exclude_dirs is None:
        exclude_dirs = []
    if exclude_files is None:
        exclude_files = []
    
    print(f"Cleaning directory: {directory}")
    
    # Delete __pycache__ directories
    for root, dirs, files in os.walk(directory, topdown=True):
        for dir_name in dirs:
            if dir_name == "__pycache__" and dir_name not in exclude_dirs:
                pycache_path = os.path.join(root, dir_name)
                print(f"  Removing Python cache directory: {pycache_path}")
                shutil.rmtree(pycache_path)
    
    # Remove redundant files in the root directory
    redundant_files = [
        # Duplicate and obsolete READMEs
        "README.md.backup",
        # Redundant setup files after GitHub push
        "prepare_for_github.py",
        "prepare_for_github_branch.py",
        # Obsolete or project-specific batch files
        "run_prediction.bat",
        # Standalone image files that should be in outputs
        "predictions.png",
        "training_history.png",
        # Model files that should be in outputs
        "best_model.pth"
    ]
    
    for file_name in redundant_files:
        if file_name not in exclude_files and os.path.exists(os.path.join(directory, file_name)):
            file_path = os.path.join(directory, file_name)
            print(f"  Removing redundant file: {file_path}")
            os.remove(file_path)
    
    # Clean up empty directories
    empty_dirs = [
        "results"  # Empty directory
    ]
    
    for dir_name in empty_dirs:
        dir_path = os.path.join(directory, dir_name)
        if os.path.exists(dir_path) and not os.listdir(dir_path):
            print(f"  Removing empty directory: {dir_path}")
            shutil.rmtree(dir_path)
    
    # Organize README files
    readme_files = {
        "GITHUB_README.md": "README.md",
        "INTRADAY_README.md": "docs/INTRADAY_README.md",
        "ENHANCED_README.md": "docs/ENHANCED_README.md"
    }
    
    # Create docs directory if it doesn't exist
    docs_dir = os.path.join(directory, "docs")
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)
    
    for src_name, dst_name in readme_files.items():
        src_path = os.path.join(directory, src_name)
        dst_path = os.path.join(directory, dst_name)
        
        if os.path.exists(src_path):
            if src_name == "GITHUB_README.md" and os.path.exists(os.path.join(directory, "README.md")):
                # If both exist, replace README.md with GITHUB_README.md
                os.remove(os.path.join(directory, "README.md"))
            
            # Create parent directory if it doesn't exist
            parent_dir = os.path.dirname(dst_path)
            if not os.path.exists(parent_dir):
                os.makedirs(parent_dir)
            
            print(f"  Moving {src_path} to {dst_path}")
            shutil.copy2(src_path, dst_path)
            os.remove(src_path)

def main():
    print("Starting project cleanup...")
    
    # Clean up the main project directory
    cleanup_directory(".")
    
    print("\nProject cleanup completed!")
    print("\nRemaining structure is now clean and organized.")

if __name__ == "__main__":
    main() 