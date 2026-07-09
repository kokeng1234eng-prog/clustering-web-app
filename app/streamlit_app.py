import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from PIL import Image

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
FEATURES_PATH = ROOT_DIR / "models" / "feature_names.json"

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    pipeline = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH, 'r') as f:
        feature_names = json.load(f)
    return pipeline, feature_names

pipeline, feature_names = load_models()

# --- IDENTIFY FEATURES ---
numeric_features = [
    'population_density', 'avg_income', 'internet_penetration',
    'avg_rent', 'air_quality_index', 'public_transport_score',
    'happiness_score', 'green_space_ratio'
]
country_dummies = [col for col in feature_names if col.startswith('country_')]
country_names = [col.replace('country_', '') for col in country_dummies]

# --- LOAD DATA ---
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

df = load_data()

# --- CLUSTER NAMES ---
cluster_names_map = {
    0: "🌿 Affluent Eco-Havens",
    1: "🏡 High-Income Suburban",
    2: "🏙️ Dense Urban Centers",
    3: "🌆 Balanced Urban Centers",
    4: "📉 Developing Centers",
    5: "🏭 Industrial Dense Cities",
    -1: "❓ Noise / Outliers"
}

# --- SECTION 1: VISUALIZATION (Static Image) ---
st.header("📊 Existing Cluster Visualization")

img_path = Path(__file__).parent / "cluster_visualization.png"
if img_path.exists():
    st.image(str(img_path), use_container_width=True)
    st.caption("UMAP projection of cities, color-coded by cluster")
else:
    st.warning("Visualization image not found. Please run the notebook to generate it.")

# --- SECTION 2: SINGLE PREDICTION ---
st.header("🔮 Predict Cluster for New City")
st.markdown("Enter city characteristics below to see which lifestyle cluster it belongs to:")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Economic & Environmental")
    avg_income = st.slider("💰 Average Income (USD)", 500, 120000, 30000, step=500)
    avg_rent = st.slider("🏠 Average Monthly Rent (USD)", 50, 6000, 800, step=50)
    air_quality_index = st.slider("🌫️ Air Quality Index (lower = better)", 0, 150, 35, step=1)
    population_density = st.slider("👥 Population Density (per km²)", 10, 25000, 3000, step=100)

with col2:
    st.subheader("Social & Infrastructure")
    internet_penetration = st.slider("📶 Internet Penetration (%)", 0, 100, 83, step=1)
    happiness_score = st.slider("😊 Happiness Score", 0.0, 10.0, 7.9, step=0.1)
    public_transport_score = st.slider("🚌 Public Transport Score", 0, 100, 60, step=1)
    green_space_ratio = st.slider("🌳 Green Space Ratio (%)", 0, 60, 33, step=1)

selected_country = st.selectbox("🌍 Country", country_names)

if st.button("🚀 Predict Cluster", type="primary"):
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
    for country in country_names:
        input_dict[f'country_{country}'] = 1 if country == selected_country else 0
    
    input_df = pd.DataFrame([input_dict])[feature_names]
    
    try:
        cluster_label = pipeline.predict(input_df)[0]
        cluster_name = cluster_names_map.get(cluster_label, f"Cluster {cluster_label}")
        
        if cluster_label == -1:
            st.warning("⚠️ This city is classified as **Noise / Outlier** – it doesn't fit neatly into any lifestyle archetype.")
        else:
            st.success(f"✅ This city belongs to: **{cluster_name}**")
            
            # Get cluster profile from cached df
            # First, assign clusters to df for profiling
            if 'cluster' not in df.columns:
                # Get labels for all data
                df_encoded = pd.get_dummies(df, columns=['country'], drop_first=True)
                X_full = df_encoded.drop(['city_name'], axis=1)
                X_full = X_full[feature_names]
                df['cluster'] = pipeline.predict(X_full)
            
            cluster_profile = df[df['cluster'] == cluster_label][numeric_features].mean()
            
            st.subheader("📋 Cluster Profile (Average Values)")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Avg Income", f"${cluster_profile['avg_income']:,.0f}")
                st.metric("Avg Rent", f"${cluster_profile['avg_rent']:,.0f}")
            with col_b:
                st.metric("Air Quality", f"{cluster_profile['air_quality_index']:.1f}")
                st.metric("Population Density", f"{cluster_profile['population_density']:,.0f}")
            with col_c:
                st.metric("Internet Penetration", f"{cluster_profile['internet_penetration']:.0f}%")
                st.metric("Happiness Score", f"{cluster_profile['happiness_score']:.1f}")
            
            st.subheader("🏙️ Cities in this cluster")
            cities = df[df['cluster'] == cluster_label]['city_name'].tolist()
            st.write(", ".join(cities[:10]) + ("..." if len(cities) > 10 else ""))
            
    except Exception as e:
        st.error(f"Error: {e}")

# --- SECTION 3: BATCH UPLOAD ---
st.header("📤 Upload Batch File")

uploaded_file = st.file_uploader("Upload CSV with city features", type=['csv'])

if uploaded_file is not None:
    batch_df = pd.read_csv(uploaded_file)
    st.write("Preview:", batch_df.head())
    
    if st.button("📊 Process Batch"):
        try:
            batch_encoded = pd.get_dummies(batch_df, columns=['country'], drop_first=True)
            for col in country_dummies:
                if col not in batch_encoded.columns:
                    batch_encoded[col] = 0
            X_batch = batch_encoded.drop(['city_name'], axis=1) if 'city_name' in batch_encoded else batch_encoded
            X_batch = X_batch[feature_names]
            
            predictions = pipeline.predict(X_batch)
            
            batch_df['predicted_cluster'] = predictions
            batch_df['cluster_name'] = batch_df['predicted_cluster'].map(cluster_names_map)
            
            st.success("✅ Batch processing complete!")
            st.dataframe(batch_df[['city_name', 'predicted_cluster', 'cluster_name']])
            
            csv = batch_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results CSV",
                data=csv,
                file_name="cluster_predictions.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Error: {e}")

st.markdown("---")
st.caption("Built with Streamlit • Clustering model: DBSCAN + KNN • Visualization: UMAP (static)")

        
