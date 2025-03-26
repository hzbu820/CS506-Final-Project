import os
import shutil
import sys
from glob import glob

def ensure_dir(dir_path):
    """Create directory if it doesn't exist"""
    os.makedirs(dir_path, exist_ok=True)

def copy_dir(src, dst, exclude_patterns=None):
    """Copy directory contents, excluding specified patterns"""
    if exclude_patterns is None:
        exclude_patterns = []
    
    ensure_dir(dst)
    if os.path.exists(src):
        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dst, item)
            
            # Check if file should be excluded
            skip = False
            for pattern in exclude_patterns:
                if pattern in s:
                    skip = True
                    break
            
            if skip:
                continue
                
            if os.path.isdir(s):
                copy_dir(s, d, exclude_patterns)
            else:
                shutil.copy2(s, d)

def create_gitkeep_files():
    """Create .gitkeep files in important directories to preserve structure"""
    dirs_to_keep = [
        'outputs/models',
        'outputs/predictions',
        'outputs/figures',
        'outputs/ensemble_analysis',
        'outputs/normalized_ensemble',
        'outputs/multiple_runs',
        'outputs/signals',
        'outputs/data',
        'outputs/ensemble',
        'outputs/ensemble_runs'
    ]
    
    for dir_path in dirs_to_keep:
        ensure_dir(dir_path)
        with open(os.path.join(dir_path, '.gitkeep'), 'w') as f:
            pass

def copy_sample_model():
    """Copy a sample model file for reference"""
    model_files = glob('outputs/models/*.pth')
    if model_files:
        # Find the smallest model file
        smallest_model = min(model_files, key=os.path.getsize)
        # Create sample_models directory
        ensure_dir('sample_models')
        # Copy the model
        sample_path = os.path.join('sample_models', 'sample_model.pth')
        shutil.copy2(smallest_model, sample_path)
        print(f"Copied sample model: {smallest_model} -> {sample_path}")
        
        # Also keep a copy in outputs/models with sample in the name
        model_sample_path = os.path.join('outputs/models', 'sample_model.pth')
        shutil.copy2(smallest_model, model_sample_path)
        print(f"Copied sample model: {smallest_model} -> {model_sample_path}")

def prepare_for_github():
    """Prepare the project for GitHub with outputs included"""
    print("Preparing LSTM Stock Prediction project for GitHub with outputs...")
    
    # Create .gitkeep files
    create_gitkeep_files()
    
    # Copy a sample model
    copy_sample_model()
    
    # Create a GitHub-specific version of the README
    if os.path.exists('README.md'):
        with open('README.md', 'r') as f:
            readme_content = f.read()
        
        # Add information about outputs
        additional_content = """
## Output Files

This repository includes output files from our experiments:

- **Figures**: Visualizations of predictions, model performance, and technical indicators
- **Predictions**: CSV files containing prediction data
- **Ensemble Analysis**: Results from our ensemble prediction approaches
- **Normalized Ensemble**: Results from normalized ensemble methods that address scaling issues

These output files demonstrate the model's performance and provide examples of the system's capabilities.
"""
        # Add the additional content before any ## sections
        if '##' in readme_content:
            pos = readme_content.find('##')
            new_readme = readme_content[:pos] + additional_content + readme_content[pos:]
        else:
            new_readme = readme_content + additional_content
        
        # Write the updated README
        with open('README.md', 'w') as f:
            f.write(new_readme)
        
        print("Updated README.md with output files information")
    
    print("\nProject is now ready for GitHub with outputs included!")
    print("\nNext steps:")
    print("1. Review the files to be committed")
    print("2. Run these git commands:")
    print("   git add .")
    print("   git commit -m \"Add LSTM Stock Prediction System with outputs\"")
    print("   git push origin your-branch-name")

if __name__ == "__main__":
    prepare_for_github() 