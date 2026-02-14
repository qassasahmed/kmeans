"""
Utility functions for K-Means clustering lab.

This module provides helper functions for:
- Data loading and preprocessing
- Visualization (elbow plots, silhouette analysis, cluster plots)
- Evaluation metrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from typing import Optional, Tuple
import os
from pathlib import Path


# Data loading functions
def load_mall_customers() -> pd.DataFrame:
    """
    Load the Mall Customers dataset.
    
    Downloads the dataset on first run using a sample or inline data.
    For production use, download from:
    https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python
    
    Returns
    -------
    df : pd.DataFrame
        Mall customers dataset with columns: CustomerID, Gender, Age,
        Annual Income (k$), Spending Score (1-100)
    """
    data_dir = Path(__file__).parent.parent / 'data'
    data_dir.mkdir(exist_ok=True)
    filepath = data_dir / 'mall_customers.csv'
    
    if filepath.exists():
        return pd.read_csv(filepath)
    
    # Create sample data if file doesn't exist
    print("Creating sample Mall Customers dataset...")
    np.random.seed(42)
    n_samples = 200
    
    data = {
        'CustomerID': range(1, n_samples + 1),
        'Gender': np.random.choice(['Male', 'Female'], size=n_samples),
        'Age': np.random.randint(18, 70, size=n_samples),
        'Annual Income (k$)': np.random.randint(15, 140, size=n_samples),
        'Spending Score (1-100)': np.random.randint(1, 100, size=n_samples)
    }
    
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    print(f"Dataset saved to {filepath}")
    
    return df


def load_sample_image() -> np.ndarray:
    """
    Load or create a sample image for compression example.
    
    Returns
    -------
    image : ndarray of shape (height, width, 3)
        RGB image array with values in range [0, 255].
    """
    from PIL import Image
    
    data_dir = Path(__file__).parent.parent / 'data'
    data_dir.mkdir(exist_ok=True)
    
    # Create a simple colorful test image
    print("Creating sample image for compression example...")
    width, height = 200, 200
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create colorful patterns
    for i in range(height):
        for j in range(width):
            image[i, j] = [
                int(255 * np.sin(i / 20) ** 2),
                int(255 * np.cos(j / 20) ** 2),
                int(255 * np.sin((i + j) / 30) ** 2)
            ]
    
    return image


# Visualization functions
def plot_clusters_2d(
    X: np.ndarray,
    labels: np.ndarray,
    centroids: Optional[np.ndarray] = None,
    title: str = "K-Means Clustering Results",
    feature_names: Optional[list] = None
) -> go.Figure:
    """
    Create a 2D scatter plot of clusters using Plotly.
    
    Parameters
    ----------
    X : ndarray of shape (n_samples, 2)
        2D data points.
    labels : ndarray of shape (n_samples,)
        Cluster labels.
    centroids : ndarray of shape (n_clusters, 2), optional
        Cluster centroids to plot.
    title : str
        Plot title.
    feature_names : list of str, optional
        Names of the two features.
    
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive Plotly figure.
    """
    if feature_names is None:
        feature_names = ['Feature 1', 'Feature 2']
    
    df = pd.DataFrame({
        feature_names[0]: X[:, 0],
        feature_names[1]: X[:, 1],
        'Cluster': labels.astype(str)
    })
    
    fig = px.scatter(
        df,
        x=feature_names[0],
        y=feature_names[1],
        color='Cluster',
        title=title,
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    
    # Add centroids if provided
    if centroids is not None:
        fig.add_trace(go.Scatter(
            x=centroids[:, 0],
            y=centroids[:, 1],
            mode='markers',
            marker=dict(
                size=15,
                color='black',
                symbol='x',
                line=dict(width=2, color='white')
            ),
            name='Centroids',
            showlegend=True
        ))
    
    fig.update_layout(
        template='plotly_white',
        width=800,
        height=600
    )
    
    return fig


def plot_clusters_3d(
    X: np.ndarray,
    labels: np.ndarray,
    centroids: Optional[np.ndarray] = None,
    title: str = "K-Means Clustering Results (3D)",
    feature_names: Optional[list] = None
) -> go.Figure:
    """
    Create a 3D scatter plot of clusters using Plotly.
    
    Parameters
    ----------
    X : ndarray of shape (n_samples, 3)
        3D data points (or PCA-reduced data).
    labels : ndarray of shape (n_samples,)
        Cluster labels.
    centroids : ndarray of shape (n_clusters, 3), optional
        Cluster centroids to plot.
    title : str
        Plot title.
    feature_names : list of str, optional
        Names of the three features/components.
    
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive 3D Plotly figure.
    """
    if feature_names is None:
        feature_names = ['PC1', 'PC2', 'PC3']
    
    df = pd.DataFrame({
        feature_names[0]: X[:, 0],
        feature_names[1]: X[:, 1],
        feature_names[2]: X[:, 2],
        'Cluster': labels.astype(str)
    })
    
    fig = px.scatter_3d(
        df,
        x=feature_names[0],
        y=feature_names[1],
        z=feature_names[2],
        color='Cluster',
        title=title,
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    
    # Add centroids if provided
    if centroids is not None:
        fig.add_trace(go.Scatter3d(
            x=centroids[:, 0],
            y=centroids[:, 1],
            z=centroids[:, 2],
            mode='markers',
            marker=dict(
                size=10,
                color='black',
                symbol='x',
                line=dict(width=2, color='white')
            ),
            name='Centroids',
            showlegend=True
        ))
    
    fig.update_layout(
        template='plotly_white',
        width=900,
        height=700
    )
    
    return fig


def plot_elbow(
    k_range: range,
    inertias: list,
    title: str = "Elbow Method for Optimal K"
) -> go.Figure:
    """
    Create an elbow plot to help determine optimal number of clusters.
    
    Parameters
    ----------
    k_range : range
        Range of K values tested.
    inertias : list
        Within-cluster sum of squares (WCSS) for each K.
    title : str
        Plot title.
    
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive Plotly figure.
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(k_range),
        y=inertias,
        mode='lines+markers',
        marker=dict(size=10, color='#1f77b4'),
        line=dict(width=3, color='#1f77b4'),
        name='WCSS'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Number of Clusters (K)',
        yaxis_title='Within-Cluster Sum of Squares (WCSS)',
        template='plotly_white',
        width=800,
        height=500,
        hovermode='x unified'
    )
    
    return fig


def plot_silhouette_scores(
    k_range: range,
    silhouette_scores: list,
    title: str = "Silhouette Score Analysis"
) -> go.Figure:
    """
    Create a silhouette score plot across different K values.
    
    Parameters
    ----------
    k_range : range
        Range of K values tested.
    silhouette_scores : list
        Average silhouette scores for each K.
    title : str
        Plot title.
    
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive Plotly figure.
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(k_range),
        y=silhouette_scores,
        mode='lines+markers',
        marker=dict(size=10, color='#2ca02c'),
        line=dict(width=3, color='#2ca02c'),
        name='Silhouette Score'
    ))
    
    # Add reference line at y=0
    fig.add_hline(
        y=0,
        line_dash='dash',
        line_color='red',
        annotation_text='Poor clustering',
        annotation_position='right'
    )
    
    fig.update_layout(
        title=title,
        xaxis_title='Number of Clusters (K)',
        yaxis_title='Average Silhouette Score',
        template='plotly_white',
        width=800,
        height=500,
        hovermode='x unified'
    )
    
    return fig


def plot_silhouette_analysis(
    X: np.ndarray,
    labels: np.ndarray,
    n_clusters: int
) -> plt.Figure:
    """
    Create a detailed silhouette analysis plot using matplotlib.
    
    Shows the silhouette coefficient for each sample, grouped by cluster.
    
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Data points.
    labels : ndarray of shape (n_samples,)
        Cluster labels.
    n_clusters : int
        Number of clusters.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Compute silhouette scores
    silhouette_avg = silhouette_score(X, labels)
    sample_silhouette_values = silhouette_samples(X, labels)
    
    y_lower = 10
    colors = plt.cm.Set2(np.linspace(0, 1, n_clusters))
    
    for i in range(n_clusters):
        # Get silhouette values for cluster i
        cluster_silhouette_values = sample_silhouette_values[labels == i]
        cluster_silhouette_values.sort()
        
        size_cluster_i = cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            cluster_silhouette_values,
            facecolor=colors[i],
            edgecolor=colors[i],
            alpha=0.7,
            label=f'Cluster {i}'
        )
        
        # Label cluster
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        
        y_lower = y_upper + 10
    
    ax.set_xlabel('Silhouette Coefficient', fontsize=12)
    ax.set_ylabel('Cluster Label', fontsize=12)
    ax.set_title(f'Silhouette Analysis (K={n_clusters}, avg={silhouette_avg:.3f})', fontsize=14)
    
    # Vertical line for average silhouette score
    ax.axvline(x=silhouette_avg, color='red', linestyle='--', linewidth=2, label=f'Average: {silhouette_avg:.3f}')
    
    ax.set_yticks([])
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


def reduce_dimensions_pca(X: np.ndarray, n_components: int = 2) -> Tuple[np.ndarray, PCA]:
    """
    Reduce dimensionality using PCA for visualization.
    
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        High-dimensional data.
    n_components : int, default=2
        Number of components to keep (2 or 3 for visualization).
    
    Returns
    -------
    X_reduced : ndarray of shape (n_samples, n_components)
        Dimensionality-reduced data.
    pca : sklearn.decomposition.PCA
        Fitted PCA object with explained variance info.
    """
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)
    
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.2%}")
    
    return X_reduced, pca


def create_comparison_figure(
    X: np.ndarray,
    labels_dict: dict,
    centroids_dict: dict = None,
    title: str = "K-Means Comparison"
) -> go.Figure:
    """
    Create side-by-side comparison of different clustering results.
    
    Parameters
    ----------
    X : ndarray of shape (n_samples, 2)
        2D data points.
    labels_dict : dict
        Dictionary mapping method names to label arrays.
    centroids_dict : dict, optional
        Dictionary mapping method names to centroid arrays.
    title : str
        Overall title.
    
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Subplot figure with comparisons.
    """
    from plotly.subplots import make_subplots
    
    n_methods = len(labels_dict)
    fig = make_subplots(
        rows=1,
        cols=n_methods,
        subplot_titles=list(labels_dict.keys())
    )
    
    colors = px.colors.qualitative.Set2
    
    for idx, (method_name, labels) in enumerate(labels_dict.items(), 1):
        # Add scatter plot for each cluster
        for cluster_id in np.unique(labels):
            mask = labels == cluster_id
            fig.add_trace(
                go.Scatter(
                    x=X[mask, 0],
                    y=X[mask, 1],
                    mode='markers',
                    marker=dict(size=5, color=colors[cluster_id % len(colors)]),
                    name=f'Cluster {cluster_id}',
                    showlegend=(idx == 1),
                    legendgroup=f'cluster{cluster_id}'
                ),
                row=1,
                col=idx
            )
        
        # Add centroids if provided
        if centroids_dict and method_name in centroids_dict:
            centroids = centroids_dict[method_name]
            fig.add_trace(
                go.Scatter(
                    x=centroids[:, 0],
                    y=centroids[:, 1],
                    mode='markers',
                    marker=dict(size=12, color='black', symbol='x', line=dict(width=2)),
                    name='Centroids',
                    showlegend=(idx == 1)
                ),
                row=1,
                col=idx
            )
    
    fig.update_layout(
        title_text=title,
        template='plotly_white',
        height=400,
        width=400 * n_methods
    )
    
    return fig
