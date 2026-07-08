# 🌆 City Lifestyle Segmentation – Clustering Web App

**Author:** [Na Kuan Ren]  
**Student ID:** [BIT_B2201F-2505003]  
**Assignment:** DA_Week10_Clustering_WebApp_Assignment

---

## 📌 Overview
This project segments global cities into lifestyle archetypes using unsupervised clustering (**HDBSCAN**) and presents the model through an interactive **Streamlit** web application. The model identifies 4-5 natural groupings based on economic, environmental, and social indicators.

---

## 📊 Dataset
- **Source:** [City Lifestyle Segmentation Dataset](https://www.kaggle.com/datasets/umuttuygurr/city-lifestyle-segmentation-dataset) (Kaggle)
- **Size:** 300 cities × 10 features (8 numeric, 2 categorical)
- **License:** Open (Kaggle public)
- **Citation:** Umut Tuygur, "City Lifestyle Segmentation Dataset," Kaggle, 2025.

---

## 🧠 Clustering Method & Justification
**Primary Algorithm:** HDBSCAN (Hierarchical Density-Based Spatial Clustering)

**Why not K-Means/GMM/PCA?**
- HDBSCAN handles clusters of varying density and irregular shapes.
- Automatically identifies noise/outliers (no need to specify `k`).
- Hierarchical structure allows multi-resolution analysis.

**Preprocessing:**
- StandardScaler for feature normalization.
- One-hot encoding for the `country` column.
- Dimensionality reduction for visualization: **UMAP** (not PCA).

---

## 📈 Evaluation Metrics
| Metric | Score |
| :--- | :--- |
| Silhouette Score | **[0.461]** |
| Davies-Bouldin Index | **[0.844]** |
| Calinski-Harabasz Index | **[170.197]** |

---

## 🏷️ Cluster Interpretation

| Cluster ID | Name | Description | Key Stats (Avg) |
| :--- | :--- | :--- | :--- |
| 0 | 🌿 **Affluent Eco-Havens** | High-income, high internet, clean air, high happiness, abundant green space. Low population density. | Income: $3,752<br>Happiness: 8.4<br>Air Quality: 38.8<br>Green Space: 37.7% |
| 1 | 🏡 **High-Income Suburban** | Very high income and internet, moderate density, good happiness and green space. Slightly less clean air than Cluster 0. | Income: $4,176<br>Happiness: 8.1<br>Air Quality: 60.9<br>Green Space: 39.0% |
| 2 | 🏙️ **Mid-Income Dense Urban** | Moderate income and internet, high density, poor air quality, moderate happiness, but good green space. | Income: $1,822<br>Happiness: 5.6<br>Air Quality: 76.0<br>Density: 4,037/km² |
| 3 | 🌆 **Balanced Urban Centers** | Medium income, good internet, moderate density, decent air, high happiness, and reasonable green space. A balanced lifestyle. | Income: $3,436<br>Happiness: 7.9<br>Air Quality: 58.5<br>Internet: 83.2% |
| 4 | 📉 **Developing Centers** | Low income, very low internet, low rent, poor air quality, low happiness, likely low green space. | Income: $882<br>Happiness: 4.1<br>Air Quality: 83.7<br>Internet: 44.8% |
| 5 | 🏭 **Industrial Dense Cities** | Very high density, low income, poor air quality (worst), low happiness, moderate internet. Likely industrial hubs. | Income: $2,510<br>Happiness: 5.7<br>Air Quality: 94.1<br>Density: 7,451/km² |
| -1 | ❓ **Noise / Outliers** | Cities that do not fit any clean archetype due to unusual combinations of features. | 7 cities flagged as noise. |

## 🗂️ Project Structure
.
├── data/ # Dataset (CSV)
├── notebooks/ # EDA and model development (.ipynb)
├── app/ # Streamlit web application
│ └── streamlit_app.py
├── models/ # Saved pipeline, UMAP reducer, feature names
│ ├── clustering_pipeline.pkl
│ ├── umap_reducer.pkl
│ └── feature_names.json
├── requirements.txt # Python dependencies
├── README.md # This file
└── .gitignore

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10 or 3.11 (3.14 is not supported)
- Git (optional, for deployment)

### 1. Clone or download this repository
```bash
git clone [https://github.com/kokeng1234eng-prog/clustering-web-app]
cd [NaKuanRen_BIT_B2201F-2505003_clustering_app]
