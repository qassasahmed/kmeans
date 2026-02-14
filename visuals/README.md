# Visuals Directory

This folder contains generated visualizations used in the README and documentation.

## Generated Files

The following files are automatically generated when running the notebooks:

- `elbow_plot.png` - Elbow method visualization
- `customer_segmentation.png` - Customer clustering 3D plot
- `image_compression.png` - Before/after image compression comparison
- `streamlit_demo.gif` - Animated demo of the Streamlit app
- `silhouette_analysis.png` - Silhouette coefficient plots

## Creating Visuals

To regenerate these visuals:

1. Run all notebooks in order (01, 02, 03)
2. Export specific plots using Plotly's `fig.write_image()` or matplotlib's `savefig()`
3. For GIFs, use screen recording tools or `imageio` library

## Note

These files are not tracked in version control initially. Generate them locally or add manually for documentation purposes.
