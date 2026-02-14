# K-Means Clustering Implementation

This directory contains the core K-Means clustering implementation and utility functions.

## Files

### `kmeans.py`
Pure NumPy implementation of K-Means clustering with:
- K-Means++ initialization for better convergence
- Lloyd's algorithm for iterative clustering
- Convergence detection
- Fit and predict methods similar to scikit-learn

**Usage:**
```python
from kmeans import KMeans

# Create and fit model
kmeans = KMeans(n_clusters=3, init='kmeans++', random_state=42)
kmeans.fit(X)

# Get results
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
inertia = kmeans.inertia_

# Predict on new data
new_labels = kmeans.predict(X_new)
```

### `utils.py`
Helper functions for visualization, data loading, and analysis:

**Visualization Functions:**
- `plot_clusters_2d()`: 2D scatter plot with Plotly
- `plot_clusters_3d()`: Interactive 3D visualization
- `plot_elbow()`: Elbow method curve
- `plot_silhouette_scores()`: Silhouette score analysis
- `plot_silhouette_analysis()`: Detailed silhouette plot
- `create_comparison_figure()`: Side-by-side comparisons

**Data Functions:**
- `load_mall_customers()`: Load/create customer dataset
- `load_sample_image()`: Load/create sample image for compression
- `reduce_dimensions_pca()`: PCA for visualization

**Usage:**
```python
from utils import plot_clusters_2d, plot_elbow, load_mall_customers

# Load data
df = load_mall_customers()

# Plot results
fig = plot_clusters_2d(X, labels, centroids)
fig.show()
```

## Integration with Notebooks

The notebooks import these modules:
```python
import sys
sys.path.append('../src')

from kmeans import KMeans
from utils import plot_clusters_2d, load_mall_customers
```

## Testing

Both modules include comprehensive docstrings and can be tested independently:

```python
# Test K-Means
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=300, centers=4, random_state=42)

kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
print(f"Inertia: {kmeans.inertia_}")
print(f"Iterations: {kmeans.n_iter_}")
```

## Dependencies

- numpy
- pandas
- matplotlib
- plotly
- scikit-learn (for metrics and comparisons)
- seaborn (for styling)
- Pillow (for image handling)
