import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import hdbscan  # <-- ADD THIS IMPORT
from pathlib import Path

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="City Lifestyle Clustering",
    page_icon="🌆",
    layout="wide"
)

st.title("🌆 City Lifestyle Segmentation")
st.markdown("**Cluster cities based on economic, environmental, and social indicators**")

# --- PATHS ---
ROOT_DIR = Path(__file__).parent.parent
DATA_PATH = ROOT_DIR / "data" / "city_lifestyle_dataset.csv"
MODEL_PATH = ROOT_DIR / "models" / "clustering_pipeline.pkl"
UMAP_PATH = ROOT_DIR / "models" / "umap_reducer.pkl"
FEATURES_PATH = ROOT_DIR / "models" / "feature_names.json"

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    pipeline = joblib.load(MODEL_PATH)
    umap_reducer = joblib.load(UMAP_PATH)
    with open(FEATURES_PATH, 'r') as f:
        feature_names = json.load(f)
    return pipeline, umap_reducer, feature_names

pipeline, umap_reducer, feature_names = load_models()

# --- IDENTIFY COLUMN TYPES ---
numeric_features = [
    'population_density', 'avg_income', 'internet_penetration',
    'avg_rent', 'air_quality_index', 'public_transport_score',
    'happiness_score', 'green_space_ratio'
]
country_dummies = [col for col in feature_names if col.startswith('country_')]
country_names = [col.replace('country_', '') for col in country_dummies]

# --- LOAD AND PREPROCESS DATA FOR VISUALIZATION ---
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv(DATA_PATH)
    
    # One-hot encode
    df_encoded = pd.get_dummies(df, columns=['country'], drop_first=True)
    X_raw = df_encoded.drop(['city_name'], axis=1)
    
    # Ensure ALL columns exist (fill missing dummies with 0)
    for col in feature_names:
        if col not in X_raw.columns:
            X_raw[col] = 0
    
    # Reorder to match training
    X_raw = X_raw[feature_names]
    
    # Scale using pipeline's scaler
    X_scaled = pipeline.named_steps['scaler'].transform(X_raw)
    
    # Get cluster labels
    labels = pipeline.named_steps['clusterer'].labels_
    
    # UMAP transform
    X_umap = umap_reducer.transform(X_scaled)
    
    # Safety check: if UMAP output has more than 2 columns, take first 2
    if X_umap.shape[1] != 2:
        X_umap = X_umap[:, :2]
    
    return df, X_umap, labels

df, X_umap, labels = load_and_preprocess_data()

# --- SECTION 1: VISUALIZE ---
st.header("📊 Existing Cluster Visualization")

# Create cluster names mapping
cluster_names_map = {
    0: "💰 Wealthy Digital Hubs",
    1: "🏙️ Dense Urban Metropolises",
    2: "🌿 Eco-Haven Cities",
    3: "📉 Developing Centers",
    -1: "❓ Noise / Outliers"
}
unique_labels = sorted(set(labels))
for label in unique_labels:
    if label == -1:
        cluster_names_map[-1] = "Noise / Outliers"
    else:
        cluster_names_map[label] = f"Cluster {label}"

import plotly.express as px

fig = px.scatter(
    x=X_umap[:, 0],
    y=X_umap[:, 1],
    color=labels,
    hover_data={'City': df['city_name'], 'Country': df['country']},
    title="City Clusters (UMAP Projection)",
    labels={'x': 'UMAP Dimension 1', 'y': 'UMAP Dimension 2'},
    color_continuous_scale='viridis'
)
fig.update_traces(
    hovertemplate="<b>%{customdata[0]}</b><br>Country: %{customdata[1]}<br>Cluster: %{marker.color}<extra></extra>"
)
st.plotly_chart(fig, use_container_width=True)

# --- SECTION 2: PREDICT NEW INPUT ---
st.header("🔮 Predict Cluster for New City")
st.markdown("Enter city characteristics and select a country:")

col1, col2 = st.columns(2)

with col1:
    avg_income = st.slider("Average Income (USD)", 1000, 120000, 30000, step=1000)
    avg_rent = st.slider("Average Monthly Rent (USD)", 100, 6000, 800, step=50)
    air_quality_index = st.slider("Air Quality Index (lower = better)", 0, 150, 35, step=1)
    population_density = st.slider("Population Density (per km²)", 50, 25000, 3000, step=100)

with col2:
    internet_penetration = st.slider("Internet Penetration (%)", 0, 100, 65, step=1)
    happiness_score = st.slider("Happiness Score", 0.0, 10.0, 6.5, step=0.1)
    public_transport_score = st.slider("Public Transport Score", 0.0, 10.0, 6.0, step=0.1)
    green_space_ratio = st.slider("Green Space Ratio (%)", 0, 60, 25, step=1)

selected_country = st.selectbox("Country", country_names if country_names else ["Unknown"])

if st.button("Predict Cluster", type="primary"):
    # Build input vector
    input_dict = {
        'population_density': population_density,
        'avg_income': avg_income,
        'internet_penetration': internet_penetration,
        'avg_rent': avg_rent,
        'air_quality_index': air_quality_index,
        'public_transport_score': public_transport_score,
        'happiness_score': happiness_score,
        'green_space_ratio': green_space_ratio
    }
    # Add one-hot encoded country
    for country in country_names:
        input_dict[f'country_{country}'] = 1 if country == selected_country else 0
    
    input_df = pd.DataFrame([input_dict])[feature_names]
    
    try:
        # Scale input
        input_scaled = pipeline.named_steps['scaler'].transform(input_df)
        
        # Predict using approximate_predict
        clusterer = pipeline.named_steps['clusterer']
        cluster_label_arr, probabilities_arr = hdbscan.approximate_predict(clusterer, input_scaled)
        
        # Extract scalar values
        cluster_label = cluster_label_arr[0]
        prob = probabilities_arr[0]
        
        # Get cluster name
        cluster_name = cluster_names_map.get(cluster_label, f"Cluster {cluster_label}")
        
        if cluster_label == -1:
            st.warning("⚠️ This city is classified as **noise/outlier**")
        else:
            st.success(f"✅ This city belongs to: **{cluster_name}**")
            st.info(f"**Prediction confidence:** {prob:.2f}")
            
            # Show cluster profile
            if 'cluster' not in df.columns:
                df['cluster'] = labels
            
            cluster_profile = df[df['cluster'] == cluster_label][numeric_features].mean()
            
            st.subheader("📋 Cluster Profile")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Avg Income", f"${cluster_profile['avg_income']:,.0f}")
                st.metric("Avg Rent", f"${cluster_profile['avg_rent']:,.0f}")
            with col_b:
                st.metric("Air Quality", f"{cluster_profile['air_quality_index']:.1f}")
                st.metric("Density", f"{cluster_profile['population_density']:,.0f}")
            with col_c:
                st.metric("Internet", f"{cluster_profile['internet_penetration']:.0f}%")
                st.metric("Happiness", f"{cluster_profile['happiness_score']:.1f}")
    except Exception as e:
        st.error(f"Prediction error: {e}")

# --- SECTION 3: BATCH UPLOAD ---
st.header("📤 Upload Batch File")

uploaded_file = st.file_uploader("Upload CSV with city features", type=['csv'])

if uploaded_file is not None:
    batch_df = pd.read_csv(uploaded_file)
    st.write("Preview:", batch_df.head())
    
    if st.button("Process Batch"):
        try:
            batch_encoded = pd.get_dummies(batch_df, columns=['country'], drop_first=True)
            for col in country_dummies:
                if col not in batch_encoded.columns:
                    batch_encoded[col] = 0
            X_batch = batch_encoded.drop(['city_name'], axis=1) if 'city_name' in batch_encoded.columns else batch_encoded
            X_batch = X_batch[feature_names]
            
            # Scale
            X_batch_scaled = pipeline.named_steps['scaler'].transform(X_batch)
            
            # Predict using approximate_predict for all points
            clusterer = pipeline.named_steps['clusterer']
            labels_batch, probs = hdbscan.approximate_predict(clusterer, X_batch_scaled)
            
            batch_df['predicted_cluster'] = labels_batch
            batch_df['cluster_name'] = batch_df['predicted_cluster'].map(cluster_names_map)
            
            st.success("Batch processing complete!")
            st.dataframe(batch_df)
            
            csv = batch_df.to_csv(index=False)
            st.download_button("📥 Download Results", csv, "cluster_predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Batch error: {e}")

st.markdown("---")
st.caption("Built with Streamlit • HDBSCAN • UMAP")
