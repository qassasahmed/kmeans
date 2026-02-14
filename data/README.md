# Data Directory

This folder contains datasets used in the K-Means clustering lab.

## Auto-Downloaded Datasets

The notebooks will automatically download the following datasets on first run:

### 1. Mall Customers Dataset
- **Source**: UCI Machine Learning Repository / Kaggle
- **Size**: ~200 rows, 5 columns
- **Features**: CustomerID, Gender, Age, Annual Income, Spending Score
- **Use Case**: Customer segmentation example
- **License**: Public domain

### 2. Sample Images
- **Source**: Built-in test images or user-provided
- **Use Case**: Image compression example
- **License**: Various (test images typically public domain)

## Dataset Citations

If you use these datasets in your research or projects, please cite:

**Mall Customers Dataset:**
```
Kaggle Mall Customer Segmentation Data
https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python
```

## Adding Your Own Data

To use your own datasets:

1. Place CSV files in this directory
2. Update `src/utils.py` to add a loader function
3. Import and use in your notebooks

**Data format for Streamlit app:**
- CSV file with numerical features
- First row = column headers
- No missing values (or handle them in preprocessing)
