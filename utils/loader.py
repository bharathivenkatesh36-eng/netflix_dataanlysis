"""
utils/loader.py
================
Centralized, cached loaders for the dataset and every pre-trained artifact
produced by `train_models.py`.

IMPORTANT: This module only ever *loads* files from disk. Nothing here calls
`.fit(...)` on a model — the Streamlit app never retrains. All heavy I/O is
wrapped in `st.cache_resource` / `st.cache_data` so each artifact is read
from disk only once per server session.
"""

from __future__ import annotations

import os

import joblib
import pandas as pd
import streamlit as st

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def _path(filename: str) -> str:
    return os.path.join(MODELS_DIR, filename)


def artifacts_exist() -> bool:
    """Whether train_models.py has been run at least once."""
    return os.path.isdir(MODELS_DIR) and os.path.exists(_path("processed_data.pkl"))


def missing_artifacts(filenames: list[str]) -> list[str]:
    """Return the subset of `filenames` that don't exist in models/ — used to
    give a friendly, specific error instead of a crash when a particular
    artifact (not just the whole models/ folder) is missing or was deleted.
    """
    return [f for f in filenames if not os.path.exists(_path(f))]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_raw_data() -> pd.DataFrame:
    return pd.read_pickle(_path("raw_data.pkl"))


@st.cache_data(show_spinner=False)
def load_processed_data() -> pd.DataFrame:
    return pd.read_pickle(_path("processed_data.pkl"))


@st.cache_data(show_spinner=False)
def load_cleaning_report() -> dict:
    return joblib.load(_path("cleaning_report.joblib"))


# ---------------------------------------------------------------------------
# Supervised learning artifacts
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_classification_models() -> dict:
    """Returns {model_name: fitted sklearn Pipeline} for all 5 trained models."""
    return joblib.load(_path("classification_model.joblib"))


@st.cache_resource(show_spinner=False)
def load_preprocessor():
    """The fitted ColumnTransformer from the best model's pipeline (for inspection)."""
    return joblib.load(_path("preprocessor.joblib"))


@st.cache_resource(show_spinner=False)
def load_scaler():
    """Standalone StandardScaler fit on the classifier's numeric features (for display)."""
    return joblib.load(_path("scaler.joblib"))


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    """Bundle: results_df, best_model_name, confusion_matrices, classification_reports,
    roc_curves, feature_importances, hyperparameter_tuning, feature_columns, excluded_features.
    """
    return joblib.load(_path("metrics.joblib"))


@st.cache_data(show_spinner=False)
def load_test_split() -> dict:
    return joblib.load(_path("test_split.joblib"))


# ---------------------------------------------------------------------------
# Unsupervised learning artifacts
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_kmeans_model():
    return joblib.load(_path("kmeans.joblib"))


@st.cache_data(show_spinner=False)
def load_cluster_results() -> dict:
    """Bundle: cluster_data, elbow_data, cluster_interpretation, cluster_scaler,
    pca_model, cluster_features.
    """
    return joblib.load(_path("cluster_results.joblib"))


# ---------------------------------------------------------------------------
# Recommendation system artifacts
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_tfidf_vectorizer():
    return joblib.load(_path("tfidf_vectorizer.joblib"))


@st.cache_resource(show_spinner=False)
def load_tfidf_matrix():
    return joblib.load(_path("tfidf_matrix.joblib"))


@st.cache_data(show_spinner=False)
def load_title_index() -> pd.Series:
    return joblib.load(_path("title_index.joblib"))


@st.cache_data(show_spinner=False)
def load_titles_metadata() -> pd.DataFrame:
    return pd.read_pickle(_path("titles_metadata.pkl"))


def recommend_titles(title: str, n: int = 10) -> tuple[pd.DataFrame | None, str | None]:
    """Return the top-n titles most similar to `title`.

    Computes cosine similarity for ONLY the queried row against the full
    TF-IDF matrix (a fast sparse matrix multiply) rather than looking up a
    pre-computed NxN matrix — see train_models.py for why that full matrix
    is deliberately not persisted. This keeps memory usage low.

    Returns (results_dataframe, error_message). Exactly one is None.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    title_index = load_title_index()
    matrix = load_tfidf_matrix()
    meta = load_titles_metadata()

    key = title.lower().strip()
    if key not in title_index.index:
        return None, f"'{title}' was not found in the dataset. Try another title."

    idx = title_index[key]
    if isinstance(idx, pd.Series):
        idx = idx.iloc[0]

    sims = cosine_similarity(matrix[idx], matrix).flatten()
    sims[idx] = -1  # exclude the title itself
    top_n = min(n, len(sims) - 1)
    top_indices = sims.argsort()[::-1][:top_n]

    results = meta.iloc[top_indices][["title", "type", "listed_in", "rating", "country", "release_year"]].copy()
    results.insert(0, "Similarity Score", [round(float(sims[i]), 3) for i in top_indices])
    results = results.rename(columns={
        "title": "Title", "type": "Type", "listed_in": "Genre",
        "rating": "Rating", "country": "Country", "release_year": "Release Year",
    })
    return results.reset_index(drop=True), None


# ---------------------------------------------------------------------------
# Insights
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_insights() -> dict:
    return joblib.load(_path("insights.joblib"))


# ---------------------------------------------------------------------------
# Live Prediction
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_inference_defaults() -> dict:
    """Small bundle: reference_year, default_number_of_genres, rating_options,
    min/max_release_year — used by the Live Prediction page to engineer a new
    title's features consistently with training. See
    `utils.preprocessing.build_live_input_features`.
    """
    return joblib.load(_path("inference_defaults.joblib"))


@st.cache_data(show_spinner=False)
def load_model_metadata() -> dict:
    """Bundle: model_version, trained_at, best_model_name, best_f1_score,
    best_cv_f1_mean, best_cv_f1_std, cv_folds, sklearn_version, training_rows.
    """
    return joblib.load(_path("model_metadata.joblib"))


@st.cache_data(show_spinner=False)
def load_explanation_stats() -> dict:
    """Per-class ({'Movie': {...}, 'TV Show': {...}}) mean of each numeric
    classifier feature, computed once at training time — used to build a
    simple, honest, rule-based prediction explanation (not SHAP).
    """
    return joblib.load(_path("explanation_stats.joblib"))
