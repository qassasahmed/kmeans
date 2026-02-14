# K-Means Clustering Lab

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5+-orange)
![Plotly](https://img.shields.io/badge/Plotly-5.0+-purple)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B)
![License](https://img.shields.io/badge/License-MIT-green)

**From theory to interactive customer segmentation in 3 notebooks** — A comprehensive, beginner-friendly guide to K-Means clustering covering mathematical foundations, from-scratch implementation, and real-world applications.

---

## Table of Contents

- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Notebooks](#-notebooks)
- [Interactive Demo](#-interactive-demo)
- [Examples](#-examples)
- [Theory Highlights](#-theory-highlights)
- [Contributing](#-contributing)
- [License](#-license)

---

## Features

- **Complete Theory Coverage**: Mathematical foundations, Lloyd's algorithm, convergence analysis
- **From-Scratch Implementation**: Pure NumPy K-Means with K-Means++ initialization
- **Beautiful Visualizations**: Interactive Plotly charts, elbow plots, silhouette analysis
- **Real-World Examples**: Customer segmentation and image compression case studies
- **Interactive Dashboard**: Streamlit app for clustering your own datasets
- **Production Comparisons**: Benchmark against scikit-learn implementation
- **Educational Focus**: Clear explanations, visual intuition, practical insights

---

## Project Structure

```
kmeans/
├── README.md                    # You are here
├── notebooks/
│   ├── 01_theory.ipynb          # K-Means fundamentals & choosing K
│   ├── 02_implementation.ipynb  # From-scratch + sklearn comparison
│   └── 03_real_world.ipynb      # Customer segmentation & image compression
├── src/
│   ├── kmeans.py                # Custom K-Means class (NumPy)
│   └── utils.py                 # Plotting & data loading helpers
├── data/                        # Auto-downloaded datasets
├── app/
│   ├── app.py                   # Streamlit interactive dashboard
│   └── requirements.txt
├── visuals/                     # Generated plots & GIFs
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## Installation

> [!IMPORTANT]
> **Prerequisites:** Python 3.10 or higher and pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/kmeans-lab.git
   cd kmeans-lab
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**
   - **Windows (PowerShell)**:
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   - **Windows (Command Prompt)**:
     ```cmd
     venv\Scripts\activate.bat
     ```
   - **macOS/Linux**:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## Quick Start

### Run the Notebooks

```bash
jupyter notebook
```

Navigate to the `notebooks/` folder and start with `01_theory.ipynb`.

### Launch the Interactive App

```bash
cd app
streamlit run app.py
```

Upload your CSV file and cluster interactively!

---

## Notebooks

### 1. [Theory Notebook](notebooks/01_theory.ipynb)
**What you'll learn:**
- What is clustering? (Unsupervised learning primer)
- Lloyd's algorithm step-by-step
- Mathematical objective function (LaTeX formatted)
- Choosing K: Elbow method & Silhouette score
- Assumptions, pitfalls, and when K-Means fails
- **Interactive animations** of centroid convergence

### 2. [Implementation Notebook](notebooks/02_implementation.ipynb)
**What you'll learn:**
- Build K-Means from scratch (pure NumPy)
- Compare with `sklearn.cluster.KMeans`
- K-Means++ vs random initialization
- Performance benchmarking
- Elbow and silhouette analysis side-by-side

### 3. [Real-World Examples](notebooks/03_real_world.ipynb)
**What you'll learn:**
- **Customer Segmentation**: Group shoppers by spending patterns (Mall Customers dataset)
- **Image Compression**: Reduce image size by 70% using color clustering
- Business insights and actionable recommendations
- 3D interactive visualizations with Plotly

---

## Interactive Demo

![Streamlit Demo](visuals/streamlit_demo.gif)

**Try it yourself:**
1. Upload any CSV with numerical features
2. Select number of clusters (K)
3. Choose features to cluster on
4. Get instant visualizations, silhouette scores, and downloadable results

---

## Examples

### Customer Segmentation
![Customer Clusters](visuals/customer_segmentation.png)

Identified 5 distinct customer groups:
- **High-Value Targets**: High income, high spending
- **Conservative Savers**: High income, low spending
- **Impulse Buyers**: Low income, high spending
- **Average Shoppers**: Mid-range on all metrics
- **Window Shoppers**: Low engagement overall

### Image Compression
![Image Compression](visuals/image_compression.png)

Reduced a 24-bit RGB image to just 16 colors using K-Means:
- **Original size**: 1.2 MB
- **Compressed size**: 0.35 MB (71% reduction)
- **Visual quality**: Perceptually similar

---

## Theory Highlights

### The K-Means Algorithm

**Objective:** Minimize within-cluster sum of squares (WCSS)

$$
\text{WCSS} = \sum_{i=1}^{K} \sum_{x \in C_i} \|x - \mu_i\|^2
$$

Where:
- $K$ = number of clusters
- $C_i$ = points in cluster $i$
- $\mu_i$ = centroid of cluster $i$

**Lloyd's Algorithm:**
1. Initialize $K$ centroids (randomly or with K-Means++)
2. **Assignment step**: Assign each point to nearest centroid
3. **Update step**: Recompute centroids as mean of assigned points
4. Repeat steps 2-3 until convergence

### Choosing K

| Method | Best For | Pros | Cons |
|--------|----------|------|------|
| **Elbow Method** | Quick visual check | Intuitive, fast | Subjective "elbow" |
| **Silhouette Score** | Validation | Quantitative metric | Slower for large data |
| **Domain Knowledge** | Business problems | Actionable | Requires expertise |

---

## Contributing

Contributions are welcome! Ideas:
- Add more real-world examples (time series, NLP, genomics)
- Implement alternative clustering algorithms (DBSCAN, hierarchical)
- Improve visualizations
- Add more evaluation metrics (Davies-Bouldin, Calinski-Harabasz)

Please open an issue or submit a pull request.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Datasets**: UCI Machine Learning Repository, scikit-learn
- **Inspiration**: FreeCodeCamp, Real Python, Business Science tutorials
- **Tools**: scikit-learn, Plotly, Streamlit, Jupyter

---

## Contact

Questions or feedback? Open an issue or reach out on [LinkedIn](https://linkedin.com/in/yourprofile).

> [!TIP]
> Star this repo if you found it helpful!
