# 🎬 Netflix Content Intelligence Dashboard

A complete, professional, end-to-end Machine Learning application built in **Streamlit**,
covering dataset analysis, preprocessing, EDA, cross-validated supervised & unsupervised ML,
model evaluation, and a memory-efficient content-based recommendation system — all wrapped in
a Netflix-inspired dark UI.

## Project Structure

```
project/
├── app.py                     # Router → pages/1_Home.py
├── train_models.py            # Offline training script — run once
├── requirements.txt
├── .streamlit/config.toml     # Netflix dark theme config
├── data/
│   └── netflix_titles.csv
├── models/                    # Saved joblib/pickle artifacts (generated)
├── images/
├── utils/
│   ├── loader.py              # Cached, read-only artifact loaders
│   ├── styling.py              # Netflix theme CSS + chart styling helpers
│   ├── preprocessing.py        # Shared cleaning/feature-engineering (used by training AND the app)
│   └── helper.py                # Small reusable utilities (formatting, top-n counts, etc.)
└── pages/
    ├── 1_Home.py
    ├── 2_Dataset_Analysis.py
    ├── 3_Data_Preprocessing.py
    ├── 4_EDA.py
    ├── 5_Supervised_ML.py
    ├── 6_Unsupervised_ML.py
    ├── 7_Recommendation_System.py
    ├── 8_Model_Performance.py
    ├── 9_About.py
    ├── 10_Insights.py          # Extra page: key findings in one place
    └── 11_Live_Prediction.py   # Predict Movie vs. TV Show on a new, unseen title
```

## Setup

```bash
cd project
pip install -r requirements.txt
```

## 1. Train the models (run once)

```bash
python train_models.py
```

This cleans the data, engineers features, trains and **cross-validates** 5 supervised
classifiers (Logistic Regression, Decision Tree, Random Forest, KNN, SVM), runs a small
**GridSearchCV** hyperparameter-tuning demo on Random Forest, runs K-Means clustering with
elbow/silhouette analysis, builds the TF-IDF recommendation engine, and saves every artifact to
`models/` with `joblib`. **The Streamlit app itself never retrains anything** — it only loads
what this script produces.

Re-run this script only if you replace `data/netflix_titles.csv` with new data.

### Saved artifacts (`models/`)

| File | Contents |
|---|---|
| `classification_model.joblib` | Dict of all 5 fitted classifier pipelines |
| `preprocessor.joblib` | Fitted `ColumnTransformer` from the best model |
| `scaler.joblib` | Standalone `StandardScaler` (numeric features, for display) |
| `metrics.joblib` | Comparison table, confusion matrices, classification reports, ROC curves, feature importances, CV & hyperparameter-tuning results |
| `kmeans.joblib` | Fitted K-Means model |
| `cluster_results.joblib` | Cluster assignments, PCA coordinates, elbow/silhouette data, cluster interpretation |
| `tfidf_vectorizer.joblib` / `tfidf_matrix.joblib` | Recommender's fitted vectorizer + sparse matrix (no NxN similarity matrix) |
| `titles_metadata.pkl` / `title_index.joblib` | Recommender lookup data |
| `insights.joblib` | Pre-computed dynamic findings for the Insights page |
| `inference_defaults.joblib` | Reference year, average genre count, and valid rating options for the Live Prediction page |
| `model_metadata.joblib` | Model version, training timestamp, CV score, best F1 — shown on the Live Prediction page |
| `explanation_stats.joblib` | Per-class feature averages, used for a simple rule-based (non-SHAP) prediction explanation |

## 2. Launch the app

```bash
streamlit run app.py
```

Pages are auto-discovered from `pages/` and appear in the sidebar in order.

## Key design decisions

- **Leakage-safe classifier.** Netflix's own genre tags (`listed_in`) literally contain the
  word "TV" for ~96% of TV Show rows and never for Movies — including it as a feature was
  verified to push every model to ~100% accuracy for the wrong reason. `duration` and its two
  derived columns are excluded for the same reason. See `utils/preprocessing.py` for the
  documented, single source of truth on which features are used vs. excluded.
- **No 8,807 × 8,807 similarity matrix on disk.** Only the compact sparse TF-IDF matrix (a few
  MB) is saved; cosine similarity for a *queried* title against all others is computed on
  demand — a sub-second sparse matrix multiply — keeping memory usage low.
- **Shared preprocessing module.** `utils/preprocessing.py` is imported by both
  `train_models.py` and the Data Preprocessing page, so the app's before/after explanation is
  guaranteed to match what the models were actually trained on.
- **Models are trained once, outside Streamlit**, with cross-validation and a hyperparameter
  tuning demo, and only ever loaded (`st.cache_resource` / `st.cache_data`) inside the app.

## Live Prediction page highlights

- **Fully dynamic model info** — best model, version, training date, CV score, and F1 are all
  read from `model_metadata.joblib`, never hardcoded.
- **Input validation** — Release Year (1888–current year), a valid Rating, and non-empty
  Country/Director/Cast/Description (20+ characters) are all checked before prediction, with
  friendly, specific error messages.
- **Staged spinners + timing** — Load Model, Engineer Features, Prediction, and Explanation are
  timed individually and summed into a Total Inference Time.
- **Confidence indicator** — 🟢 High (90–100%) / 🟡 Medium (70–89%) / 🔴 Low (<70%).
- **Rule-based explanation** — compares the new title's engineered features to each class's
  historical average (`explanation_stats.joblib`); explicitly NOT SHAP or any per-instance
  attribution.
- **Prediction history** — kept in `st.session_state` for the current session, with a Clear
  History button.
- **CSV & PDF export** — download any single prediction as a CSV row or a styled one-page PDF
  report (via `reportlab`).
- **Reset button** — clears the form back to defaults.
- **Graceful failure handling** — missing artifact files or corrupted joblib data show a
  specific, friendly error instead of crashing the page.

## Dataset

`netflix_titles.csv` — 8,807 Netflix titles (movies & TV shows) with metadata: cast,
director, country, date added, release year, rating, duration, genres, and description.
