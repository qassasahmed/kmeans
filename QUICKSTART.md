# Quick Start Guide

Get up and running with the K-Means Clustering Lab in minutes!

## Prerequisites

- Python 3.10 or higher
- pip package manager
- Git (if cloning from GitHub)

## Setup Instructions

### 1. Navigate to Project Directory

```powershell
cd c:\Users\qassas\source\repos\kmeans
```

### 2. Create Virtual Environment

```powershell
python -m venv venv
```

### 3. Activate Virtual Environment

**PowerShell:**
```powershell
.\venv\Scripts\Activate.ps1
```

**Command Prompt:**
```cmd
venv\Scripts\activate.bat
```

### 4. Install Dependencies

```powershell
pip install -r requirements.txt
```

This will install:
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- plotly
- jupyter
- ipywidgets
- Pillow
- tqdm

### 5. Launch Jupyter Notebooks

```powershell
jupyter notebook
```

Your browser should open automatically. Navigate to the `notebooks/` folder and start with `01_theory.ipynb`.

## Running the Streamlit App

### Install Streamlit Dependencies

```powershell
cd app
pip install -r requirements.txt
```

### Launch the App

```powershell
streamlit run app.py
```

The app will open in your browser at http://localhost:8501

## Notebook Execution Order

1. **01_theory.ipynb** - Learn the fundamentals
2. **02_implementation.ipynb** - Build from scratch
3. **03_real_world.ipynb** - Apply to real problems

## Troubleshooting

### Import Errors in Notebooks

If you see "Import 'kmeans' could not be resolved":
1. Make sure your virtual environment is activated
2. Run all cells in order from top to bottom
3. Check that `sys.path.append('../src')` is executed

### Jupyter Not Opening

```powershell
# Try specifying the port explicitly
jupyter notebook --port=8888
```

### Streamlit Not Working

```powershell
# Reinstall streamlit
pip uninstall streamlit
pip install streamlit --upgrade
```

### Package Installation Issues

```powershell
# Update pip first
python -m pip install --upgrade pip

# Then try installing again
pip install -r requirements.txt
```

## Verify Installation

Test that everything works:

```python
# Open Python in your activated virtual environment
python

# Test imports
>>> import numpy as np
>>> import pandas as pd
>>> import sklearn
>>> import plotly
>>> print("✅ All packages installed successfully!")
```

## Next Steps

- ✅ Run the notebooks in order
- 🎨 Try the Streamlit app with your own CSV data
- 📊 Experiment with different datasets
- 🔧 Modify the K-Means implementation
- 🚀 Share your results!

## Need Help?

- Check the [README.md](README.md) for detailed project documentation
- Review individual README files in `src/`, `data/`, and `visuals/` folders
- Check notebook markdown cells for explanations
- Open an issue on GitHub

---

**Happy clustering! 🎯**
