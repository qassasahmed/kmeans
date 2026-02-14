"""
K-Means Clustering Implementation from Scratch

This module implements the K-Means clustering algorithm using pure NumPy,
including K-Means++ initialization for better centroid selection.
"""

import numpy as np
from typing import Optional, Tuple


class KMeans:
    """
    K-Means clustering algorithm with K-Means++ initialization.
    
    Parameters
    ----------
    n_clusters : int, default=3
        The number of clusters to form.
    max_iters : int, default=300
        Maximum number of iterations of the K-Means algorithm.
    tol : float, default=1e-4
        Relative tolerance for convergence (change in inertia).
    init : {'kmeans++', 'random'}, default='kmeans++'
        Method for initialization:
        - 'kmeans++': selects initial centroids using K-Means++ algorithm
        - 'random': randomly selects initial centroids
    random_state : int or None, default=None
        Seed for random number generator for reproducibility.
    
    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers.
    labels_ : ndarray of shape (n_samples,)
        Labels of each point.
    inertia_ : float
        Sum of squared distances of samples to their closest cluster center.
    n_iter_ : int
        Number of iterations run.
    
    Examples
    --------
    >>> from sklearn.datasets import make_blobs
    >>> X, _ = make_blobs(n_samples=300, centers=4, random_state=42)
    >>> kmeans = KMeans(n_clusters=4, random_state=42)
    >>> kmeans.fit(X)
    >>> print(kmeans.labels_)
    >>> print(kmeans.cluster_centers_)
    """
    
    def __init__(
        self,
        n_clusters: int = 3,
        max_iters: int = 300,
        tol: float = 1e-4,
        init: str = 'kmeans++',
        random_state: Optional[int] = None
    ):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.init = init
        self.random_state = random_state
        
        # Attributes set during fit
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0
    
    def _initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        """
        Initialize centroids using specified method.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        
        Returns
        -------
        centroids : ndarray of shape (n_clusters, n_features)
            Initial centroids.
        """
        n_samples, n_features = X.shape
        rng = np.random.RandomState(self.random_state)
        
        if self.init == 'random':
            # Randomly select n_clusters samples as initial centroids
            indices = rng.choice(n_samples, size=self.n_clusters, replace=False)
            return X[indices].copy()
        
        elif self.init == 'kmeans++':
            # K-Means++ initialization
            centroids = np.zeros((self.n_clusters, n_features))
            
            # Choose first centroid randomly
            centroids[0] = X[rng.choice(n_samples)]
            
            # Choose remaining centroids with probability proportional to distance squared
            for i in range(1, self.n_clusters):
                # Compute distances from points to nearest existing centroid
                distances = np.min(
                    np.linalg.norm(X[:, np.newaxis] - centroids[:i], axis=2),
                    axis=1
                )
                
                # Square the distances for probability weighting
                distances_squared = distances ** 2
                
                # Avoid division by zero
                if distances_squared.sum() == 0:
                    probabilities = np.ones(n_samples) / n_samples
                else:
                    probabilities = distances_squared / distances_squared.sum()
                
                # Choose next centroid
                centroids[i] = X[rng.choice(n_samples, p=probabilities)]
            
            return centroids
        
        else:
            raise ValueError(f"init should be 'kmeans++' or 'random', got {self.init}")
    
    def _assign_labels(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Assign each sample to the nearest centroid.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        centroids : ndarray of shape (n_clusters, n_features)
            Current centroids.
        
        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the centroid each sample belongs to.
        """
        # Calculate distances from each point to each centroid
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        
        # Assign each point to nearest centroid
        return np.argmin(distances, axis=1)
    
    def _update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Compute new centroids as the mean of assigned points.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        labels : ndarray of shape (n_samples,)
            Current cluster assignments.
        
        Returns
        -------
        centroids : ndarray of shape (n_clusters, n_features)
            Updated centroids.
        """
        n_features = X.shape[1]
        centroids = np.zeros((self.n_clusters, n_features))
        
        for k in range(self.n_clusters):
            # Get all points assigned to cluster k
            cluster_points = X[labels == k]
            
            if len(cluster_points) > 0:
                # Compute mean of points in cluster
                centroids[k] = cluster_points.mean(axis=0)
            else:
                # If cluster is empty, keep previous centroid or reinitialize
                # For simplicity, we keep the previous position
                pass
        
        return centroids
    
    def _compute_inertia(self, X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
        """
        Compute the within-cluster sum of squares (WCSS).
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        labels : ndarray of shape (n_samples,)
            Cluster assignments.
        centroids : ndarray of shape (n_clusters, n_features)
            Cluster centroids.
        
        Returns
        -------
        inertia : float
            Sum of squared distances of samples to their closest cluster center.
        """
        inertia = 0.0
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - centroids[k]) ** 2)
        return inertia
    
    def fit(self, X: np.ndarray) -> 'KMeans':
        """
        Compute K-Means clustering.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        
        Returns
        -------
        self : KMeans
            Fitted estimator.
        """
        n_samples, n_features = X.shape
        
        if self.n_clusters > n_samples:
            raise ValueError(
                f"n_clusters={self.n_clusters} must be <= n_samples={n_samples}"
            )
        
        # Initialize centroids
        centroids = self._initialize_centroids(X)
        
        # Lloyd's algorithm
        prev_inertia = np.inf
        
        for iteration in range(self.max_iters):
            # Assignment step: assign points to nearest centroid
            labels = self._assign_labels(X, centroids)
            
            # Update step: recompute centroids
            new_centroids = self._update_centroids(X, labels)
            
            # Compute inertia
            inertia = self._compute_inertia(X, labels, new_centroids)
            
            # Check for convergence
            if np.abs(prev_inertia - inertia) < self.tol:
                self.n_iter_ = iteration + 1
                break
            
            centroids = new_centroids
            prev_inertia = inertia
        else:
            self.n_iter_ = self.max_iters
        
        # Store final results
        self.cluster_centers_ = centroids
        self.labels_ = labels
        self.inertia_ = inertia
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the closest cluster each sample in X belongs to.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            New data to predict.
        
        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        if self.cluster_centers_ is None:
            raise ValueError("Model must be fitted before calling predict()")
        
        return self._assign_labels(X, self.cluster_centers_)
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Compute cluster centers and predict cluster index for each sample.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        
        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        self.fit(X)
        return self.labels_
