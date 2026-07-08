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

| Cluster ID | Name | Description |
| :--- | :--- | :--- |
| -1 | ❌ Noise / Outliers | 14 cities with conflicting features – high density but very low income, poor air quality (86.6), low happiness (5.6). Don't fit any clean archetype. |
| 0 | 🌿 Eco-Haven Cities | The greenest and happiest cities! Lowest air pollution (38.8), highest green space (37.7%), highest happiness (8.40), low density (1,163/km²). Residents enjoy clean, spacious living. |
| 1 | 💰 Wealthy Digital Hubs | The economic powerhouses – highest income ($4,176), highest rent ($1,479), and highest internet penetration (88.5%). High happiness (8.14) and moderate density (2,460/km²). |
| 2 | 📉 Developing Urban Centers | Low income ($1,822), low rent ($644), low happiness (5.60), poor air quality (76.0). Moderate density (4,037/km²) suggests crowded but under-resourced cities. |
| 3 | 🏙️ Balanced Metropolises | The "middle ground" – good income ($3,436), high happiness (7.91), high internet (83.2%), moderate density (3,048/km²). Well-rounded urban lifestyle. |
| 4 | 📉 Struggling Cities | The lowest of everything – lowest income ($882), lowest happiness (4.08), lowest internet (44.8%), lowest green space (~18%), poorest air quality (83.7). Cities facing significant challenges. |
| 5 | 🏭 Industrial Mega-Cities | Very high density (7,451/km²) with the worst air quality (94.1), low income ($2,510), and low happiness (5.69). Likely heavily industrialized urban centers. |
---

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
