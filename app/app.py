"""
K-Means Clustering Interactive Dashboard

A Streamlit app for interactive K-Means clustering on user-uploaded CSV data.
Upload your own dataset and explore clusters in real-time!
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import io

# Page configuration
st.set_page_config(
    page_title="K-Means Clustering Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🎯 K-Means Clustering Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload your data and discover hidden patterns</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("📁 Data Upload")
uploaded_file = st.sidebar.file_uploader(
    "Choose a CSV file",
    type=['csv'],
    help="Upload a CSV file with numerical features"
)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Clustering Parameters")

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'features' not in st.session_state:
    st.session_state.features = []

# Load data
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state.data = df
        
        # Success message
        st.sidebar.success(f"✅ Loaded {len(df)} rows × {len(df.columns)} columns")
        
    except Exception as e:
        st.sidebar.error(f"Error loading file: {e}")
        st.stop()
else:
    # Show instructions
    st.info("👈 **Get started:** Upload a CSV file using the sidebar")
    
    st.markdown("""
    ### 📋 Instructions
    
    1. **Upload your data**: Click "Browse files" in the sidebar
    2. **Select features**: Choose which columns to use for clustering
    3. **Set parameters**: Adjust the number of clusters (K)
    4. **Run clustering**: Click "Run K-Means" button
    5. **Analyze results**: Explore visualizations and download clustered data
    
    ### 📊 Data Requirements
    
    - CSV format with column headers
    - At least 2 numerical columns
    - No missing values (or handle them first)
    - Ideally 50+ rows for meaningful clusters
    
    ### 💡 Example Use Cases
    
    - **Customer Segmentation**: Age, Income, Spending
    - **Market Analysis**: Product features, prices, ratings
    - **Sensor Data**: Temperature, pressure, humidity readings
    - **Image Features**: Color channels, texture metrics
    """)
    
    st.stop()

# Main content when data is loaded
df = st.session_state.data

# Show data preview
with st.expander("📊 Data Preview", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", len(df))
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        st.metric("Numerical Columns", len(numeric_cols))
    
    st.dataframe(df.head(10), use_container_width=True)

# Feature selection
st.sidebar.subheader("📌 Select Features")

# Get numerical columns
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

if len(numeric_columns) < 2:
    st.error("❌ Dataset must have at least 2 numerical columns for clustering!")
    st.stop()

selected_features = st.sidebar.multiselect(
    "Choose features for clustering:",
    options=numeric_columns,
    default=numeric_columns[:min(3, len(numeric_columns))],
    help="Select 2 or more numerical features"
)

if len(selected_features) < 2:
    st.warning("⚠️ Please select at least 2 features from the sidebar")
    st.stop()

# Clustering parameters
st.sidebar.subheader("🎛️ K-Means Settings")

n_clusters = st.sidebar.slider(
    "Number of Clusters (K)",
    min_value=2,
    max_value=min(10, len(df) - 1),
    value=3,
    help="Choose how many groups to create"
)

init_method = st.sidebar.selectbox(
    "Initialization Method",
    options=['k-means++', 'random'],
    index=0,
    help="k-means++ is usually better"
)

scale_data = st.sidebar.checkbox(
    "Standardize Features",
    value=True,
    help="Recommended: Scale features to mean=0, std=1"
)

# Run button
run_clustering = st.sidebar.button("🚀 Run K-Means", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 About")
st.sidebar.markdown("""
This app uses **scikit-learn's K-Means** algorithm.

**Made with:**
- Streamlit
- scikit-learn
- Plotly
- pandas

[📖 Learn More](https://github.com/yourusername/kmeans-lab)
""")

# Run clustering
if run_clustering:
    with st.spinner("🔄 Running K-Means clustering..."):
        try:
            # Prepare data
            X = df[selected_features].values
            
            # Check for missing values
            if np.isnan(X).any():
                st.error("❌ Selected features contain missing values. Please clean your data first.")
                st.stop()
            
            # Standardize if requested
            if scale_data:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
            else:
                X_scaled = X
            
            # Apply K-Means
            kmeans = KMeans(
                n_clusters=n_clusters,
                init=init_method,
                random_state=42,
                n_init=10
            )
            
            labels = kmeans.fit_predict(X_scaled)
            
            # Add cluster labels to dataframe
            df['Cluster'] = labels
            
            # Calculate metrics
            inertia = kmeans.inertia_
            silhouette = silhouette_score(X_scaled, labels)
            
            # Store results in session state
            st.session_state.clustered_data = df
            st.session_state.kmeans = kmeans
            st.session_state.X_scaled = X_scaled
            st.session_state.labels = labels
            st.session_state.inertia = inertia
            st.session_state.silhouette = silhouette
            
            st.success("✅ Clustering complete!")
            
        except Exception as e:
            st.error(f"❌ Error during clustering: {e}")
            st.stop()

# Display results if clustering has been run
if 'clustered_data' in st.session_state:
    df_clustered = st.session_state.clustered_data
    kmeans = st.session_state.kmeans
    X_scaled = st.session_state.X_scaled
    labels = st.session_state.labels
    
    st.markdown("---")
    st.header("📊 Clustering Results")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Number of Clusters",
            n_clusters,
            help="The K value you selected"
        )
    
    with col2:
        st.metric(
            "Inertia (WCSS)",
            f"{st.session_state.inertia:.2f}",
            help="Lower is better (within-cluster sum of squares)"
        )
    
    with col3:
        st.metric(
            "Silhouette Score",
            f"{st.session_state.silhouette:.3f}",
            help="Range: -1 to 1 (higher is better)"
        )
    
    with col4:
        st.metric(
            "Iterations",
            kmeans.n_iter_,
            help="Number of iterations until convergence"
        )
    
    # Cluster distribution
    st.subheader("📈 Cluster Distribution")
    
    cluster_counts = df_clustered['Cluster'].value_counts().sort_index()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Cluster Sizes:**")
        for cluster, count in cluster_counts.items():
            percentage = (count / len(df_clustered)) * 100
            st.write(f"Cluster {cluster}: {count} samples ({percentage:.1f}%)")
    
    with col2:
        fig_dist = px.bar(
            x=cluster_counts.index,
            y=cluster_counts.values,
            labels={'x': 'Cluster', 'y': 'Number of Samples'},
            title='Samples per Cluster',
            color=cluster_counts.index,
            color_continuous_scale='Set2'
        )
        fig_dist.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig_dist, use_container_width=True)
    
    # Visualizations
    st.subheader("🎨 Cluster Visualizations")
    
    # 2D scatter plot
    if len(selected_features) >= 2:
        st.markdown("**2D Scatter Plot**")
        
        # Let user choose which features to plot
        col1, col2 = st.columns(2)
        with col1:
            x_feature = st.selectbox("X-axis feature:", selected_features, index=0)
        with col2:
            y_feature = st.selectbox("Y-axis feature:", selected_features, 
                                    index=min(1, len(selected_features)-1))
        
        fig_2d = px.scatter(
            df_clustered,
            x=x_feature,
            y=y_feature,
            color='Cluster',
            title=f'{y_feature} vs {x_feature}',
            color_continuous_scale='Set2',
            hover_data=selected_features
        )
        
        # Add centroids (need to inverse transform if scaled)
        if scale_data:
            centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)
        else:
            centroids_original = kmeans.cluster_centers_
        
        x_idx = selected_features.index(x_feature)
        y_idx = selected_features.index(y_feature)
        
        fig_2d.add_trace(go.Scatter(
            x=centroids_original[:, x_idx],
            y=centroids_original[:, y_idx],
            mode='markers',
            marker=dict(size=15, color='black', symbol='x', line=dict(width=2, color='white')),
            name='Centroids',
            showlegend=True
        ))
        
        fig_2d.update_layout(height=500)
        st.plotly_chart(fig_2d, use_container_width=True)
    
    # 3D plot if 3+ features selected
    if len(selected_features) >= 3:
        st.markdown("**3D Scatter Plot**")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            x3_feature = st.selectbox("X-axis:", selected_features, index=0, key='x3')
        with col2:
            y3_feature = st.selectbox("Y-axis:", selected_features, index=1, key='y3')
        with col3:
            z3_feature = st.selectbox("Z-axis:", selected_features, index=2, key='z3')
        
        fig_3d = px.scatter_3d(
            df_clustered,
            x=x3_feature,
            y=y3_feature,
            z=z3_feature,
            color='Cluster',
            title=f'3D Cluster Visualization',
            color_continuous_scale='Set2',
            hover_data=selected_features
        )
        
        fig_3d.update_layout(height=600)
        st.plotly_chart(fig_3d, use_container_width=True)
    
    # PCA visualization if many features
    if len(selected_features) > 3:
        st.markdown("**PCA Visualization (Dimensionality Reduction)**")
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        df_pca = pd.DataFrame({
            'PC1': X_pca[:, 0],
            'PC2': X_pca[:, 1],
            'Cluster': labels
        })
        
        fig_pca = px.scatter(
            df_pca,
            x='PC1',
            y='PC2',
            color='Cluster',
            title=f'PCA Projection (Explained Variance: {pca.explained_variance_ratio_.sum():.1%})',
            color_continuous_scale='Set2'
        )
        
        fig_pca.update_layout(height=500)
        st.plotly_chart(fig_pca, use_container_width=True)
        
        st.info(f"ℹ️ PCA reduced {len(selected_features)} features to 2 dimensions. "
                f"Variance explained: PC1={pca.explained_variance_ratio_[0]:.1%}, "
                f"PC2={pca.explained_variance_ratio_[1]:.1%}")
    
    # Cluster profiles
    st.subheader("📋 Cluster Profiles")
    
    cluster_profiles = df_clustered.groupby('Cluster')[selected_features].mean().round(3)
    cluster_profiles['Count'] = df_clustered.groupby('Cluster').size()
    
    st.dataframe(cluster_profiles, use_container_width=True)
    
    # Heatmap of cluster profiles
    fig_heatmap = px.imshow(
        cluster_profiles[selected_features].T,
        labels=dict(x="Cluster", y="Feature", color="Value"),
        title="Cluster Profile Heatmap",
        color_continuous_scale='RdBu_r',
        aspect='auto'
    )
    fig_heatmap.update_layout(height=400)
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Download results
    st.subheader("💾 Download Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Download clustered data as CSV
        csv_buffer = io.StringIO()
        df_clustered.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        st.download_button(
            label="📥 Download Clustered Data (CSV)",
            data=csv_data,
            file_name="clustered_data.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Download cluster summary
        summary_buffer = io.StringIO()
        cluster_profiles.to_csv(summary_buffer)
        summary_data = summary_buffer.getvalue()
        
        st.download_button(
            label="📥 Download Cluster Summary (CSV)",
            data=summary_data,
            file_name="cluster_summary.csv",
            mime="text/csv",
            use_container_width=True
        )
