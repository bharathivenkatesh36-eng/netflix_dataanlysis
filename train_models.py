"""
train_models.py
================
Offline training script for the Netflix Content Intelligence Dashboard.

Run this ONCE (outside Streamlit) to clean the data, engineer features,
train and cross-validate every supervised classifier, run K-Means clustering,
build the TF-IDF recommendation engine, and persist every artifact with
joblib/pickle to `models/`.

The Streamlit app NEVER retrains anything — every page only loads what this
script produces. Re-run this script only if `data/netflix_titles.csv` changes.

Usage:
    python train_models.py
"""

from __future__ import annotations

import os
import time
import warnings
from datetime import datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from utils.preprocessing import (
    CLASSIFIER_CATEGORICAL_FEATURES,
    CLASSIFIER_FEATURE_COLUMNS,
    CLASSIFIER_NUMERIC_FEATURES,
    CLASSIFIER_TEXT_FEATURE,
    CLUSTER_FEATURES,
    clean_data,
    engineer_features,
)

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
DATA_PATH = "data/netflix_titles.csv"
MODELS_DIR = "models"
CV_FOLDS = 5
MODEL_VERSION = "1.0.0"  # bump manually when the pipeline/feature set changes meaningfully

os.makedirs(MODELS_DIR, exist_ok=True)


def log(msg: str) -> None:
    print(f"[train_models] {msg}")


def save(obj: Any, filename: str) -> None:
    joblib.dump(obj, os.path.join(MODELS_DIR, filename))


# ---------------------------------------------------------------------------
# 1. LOAD + CLEAN + ENGINEER  (shared functions — see utils/preprocessing.py)
# ---------------------------------------------------------------------------
log("Loading raw dataset ...")
raw_df = pd.read_csv(DATA_PATH)
raw_df.to_pickle(os.path.join(MODELS_DIR, "raw_data.pkl"))
log(f"Raw shape: {raw_df.shape}")

log("Cleaning data ...")
cleaned_df, cleaning_report = clean_data(raw_df)

log("Engineering features ...")
df = engineer_features(cleaned_df)
df.to_pickle(os.path.join(MODELS_DIR, "processed_data.pkl"))

reference_year = int(df["release_year"].max())  # recomputed here to match engineer_features()

cleaning_report["processed_shape"] = df.shape
save(cleaning_report, "cleaning_report.joblib")
log(f"Processed shape: {df.shape} | duplicates removed: {cleaning_report['duplicates_removed']}")

# ---------------------------------------------------------------------------
# 2. SUPERVISED LEARNING — Predict Movie vs TV Show (leakage-safe)
# ---------------------------------------------------------------------------
log("Preparing leakage-safe supervised learning feature set ...")
log(f"  features used: {CLASSIFIER_FEATURE_COLUMNS}")
log("  features EXCLUDED to avoid leakage: listed_in, duration, duration_minutes, number_of_seasons")

df["target"] = (df["type"] == "TV Show").astype(int)
X = df[CLASSIFIER_FEATURE_COLUMNS].copy()
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), CLASSIFIER_NUMERIC_FEATURES),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CLASSIFIER_CATEGORICAL_FEATURES),
    ("desc_tfidf", TfidfVectorizer(max_features=200, stop_words="english"), CLASSIFIER_TEXT_FEATURE),
])
save(preprocessor, "preprocessor_template.joblib")  # unfitted template, for reference/display only

candidate_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(max_depth=12, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(n_neighbors=9),
    "SVM": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
}

cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

fitted_pipelines: dict[str, Pipeline] = {}
metrics_rows: list[dict[str, Any]] = []
confusion_matrices: dict[str, np.ndarray] = {}
classification_reports: dict[str, dict] = {}
roc_curves: dict[str, dict] = {}

for name, model in candidate_models.items():
    log(f"  training {name} ...")
    pipe = Pipeline([("preprocessor", preprocessor), ("classifier", model)])

    # Cross-validation on the training split (F1 score)
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)

    train_start = time.perf_counter()
    pipe.fit(X_train, y_train)
    training_time = time.perf_counter() - train_start

    predict_start = time.perf_counter()
    preds = pipe.predict(X_test)
    prediction_time = (time.perf_counter() - predict_start) / len(X_test)  # per-sample

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    roc_auc = np.nan
    if hasattr(pipe, "predict_proba"):
        try:
            proba = pipe.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, proba)
            roc_auc = roc_auc_score(y_test, proba)
            roc_curves[name] = {"fpr": fpr, "tpr": tpr, "auc": roc_auc}
        except Exception:
            pass

    fitted_pipelines[name] = pipe
    confusion_matrices[name] = confusion_matrix(y_test, preds)
    classification_reports[name] = classification_report(
        y_test, preds, target_names=["Movie", "TV Show"], output_dict=True
    )
    metrics_rows.append({
        "Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1 Score": f1,
        "ROC AUC": roc_auc, "CV F1 Mean": cv_scores.mean(), "CV F1 Std": cv_scores.std(),
        "Training Time (s)": training_time, "Prediction Time (s/sample)": prediction_time,
    })

results_df = pd.DataFrame(metrics_rows).sort_values("F1 Score", ascending=False).reset_index(drop=True)
best_model_name = results_df.iloc[0]["Model"]
log(f"Best model by F1 score: {best_model_name}")

# ---------------------------------------------------------------------------
# 2b. Light hyperparameter tuning demo (Random Forest) — optional per spec,
#     included to demonstrate the technique without an expensive full search.
# ---------------------------------------------------------------------------
log("Running a small hyperparameter search on Random Forest (demo) ...")
rf_param_grid = {
    "classifier__n_estimators": [150, 300],
    "classifier__max_depth": [None, 20],
}
rf_pipe = Pipeline([("preprocessor", preprocessor), ("classifier", RandomForestClassifier(random_state=RANDOM_STATE))])
grid_search = GridSearchCV(rf_pipe, rf_param_grid, cv=3, scoring="f1", n_jobs=-1)
grid_search.fit(X_train, y_train)
hyperparameter_tuning_report = {
    "model": "Random Forest",
    "param_grid": rf_param_grid,
    "best_params": grid_search.best_params_,
    "best_cv_f1": grid_search.best_score_,
}
log(f"  best params: {grid_search.best_params_} (CV F1 = {grid_search.best_score_:.4f})")

# Feature importance (tree-based models only)
feature_importances: dict[str, pd.Series] = {}
try:
    feature_names = (
        CLASSIFIER_NUMERIC_FEATURES
        + list(fitted_pipelines["Random Forest"].named_steps["preprocessor"]
               .named_transformers_["cat"].get_feature_names_out(CLASSIFIER_CATEGORICAL_FEATURES))
        + [f"desc_{i}" for i in range(200)]
    )
    importances = fitted_pipelines["Random Forest"].named_steps["classifier"].feature_importances_
    n = min(len(feature_names), len(importances))
    feature_importances["Random Forest"] = pd.Series(
        importances[:n], index=feature_names[:n]
    ).sort_values(ascending=False).head(15)
except Exception as exc:
    log(f"  feature importance skipped: {exc}")

# ---------------------------------------------------------------------------
# Save supervised-learning artifacts
# ---------------------------------------------------------------------------
save(fitted_pipelines, "classification_model.joblib")  # dict[name] -> fitted Pipeline (incl. best)
save(fitted_pipelines[best_model_name].named_steps["preprocessor"], "preprocessor.joblib")  # fitted, best model's
save(StandardScaler().fit(df[CLASSIFIER_NUMERIC_FEATURES]), "scaler.joblib")  # standalone numeric scaler, for reference/display

metrics_bundle = {
    "results_df": results_df,
    "best_model_name": best_model_name,
    "confusion_matrices": confusion_matrices,
    "classification_reports": classification_reports,
    "roc_curves": roc_curves,
    "feature_importances": feature_importances,
    "hyperparameter_tuning": hyperparameter_tuning_report,
    "cv_folds": CV_FOLDS,
    "feature_columns": CLASSIFIER_FEATURE_COLUMNS,
    "excluded_features": ["listed_in", "duration", "duration_minutes", "number_of_seasons"],
}
save(metrics_bundle, "metrics.joblib")
save({"X_test": X_test.reset_index(drop=True), "y_test": y_test.reset_index(drop=True)}, "test_split.joblib")

# ---------------------------------------------------------------------------
# 2c. Model metadata — for the Live Prediction page's "Model Information" card
# ---------------------------------------------------------------------------
log("Saving model metadata ...")
best_row_for_metadata = results_df.iloc[0]  # results_df is already sorted by F1 Score descending
model_metadata = {
    "model_version": MODEL_VERSION,
    "trained_at": datetime.now().isoformat(timespec="seconds"),
    "best_model_name": best_model_name,
    "best_f1_score": float(best_row_for_metadata["F1 Score"]),
    "best_cv_f1_mean": float(best_row_for_metadata["CV F1 Mean"]),
    "best_cv_f1_std": float(best_row_for_metadata["CV F1 Std"]),
    "cv_folds": CV_FOLDS,
    "sklearn_version": sklearn.__version__,
    "training_rows": int(len(df)),
}
save(model_metadata, "model_metadata.joblib")

# ---------------------------------------------------------------------------
# 2d. Explanation stats — per-class feature averages for a simple, honest,
#     rule-based prediction explanation on the Live Prediction page (NOT
#     SHAP or any per-instance gradient attribution — just comparing a new
#     title's engineered feature values to each class's historical average,
#     computed once here and reused unchanged at inference time).
# ---------------------------------------------------------------------------
log("Saving explanation stats (per-class feature averages) ...")
explanation_stats = df.groupby("type")[CLASSIFIER_NUMERIC_FEATURES].mean().to_dict(orient="index")
save(explanation_stats, "explanation_stats.joblib")

# ---------------------------------------------------------------------------
# 3. UNSUPERVISED LEARNING — K-Means clustering
# ---------------------------------------------------------------------------
log("Running unsupervised clustering ...")

cluster_input = df[CLUSTER_FEATURES].copy()
cluster_scaler = StandardScaler()
cluster_scaled = cluster_scaler.fit_transform(cluster_input)

inertias, silhouettes = [], []
k_range = range(2, 11)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(cluster_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(cluster_scaled, labels))

# k=2 mathematically maximizes silhouette here but collapses into one huge
# "everything" cluster and one tiny outlier group — not interpretable or
# presentable. The best k in the k=3..8 window is chosen instead, balancing
# cohesion against a usable number of segments.
k_list = list(k_range)
searchable = [(k, s) for k, s in zip(k_list, silhouettes) if 3 <= k <= 8]
optimal_k = max(searchable, key=lambda pair: pair[1])[0]
log(f"Optimal k (max silhouette, 3<=k<=8): {optimal_k}")

final_kmeans = KMeans(n_clusters=optimal_k, random_state=RANDOM_STATE, n_init=10)
cluster_labels = final_kmeans.fit_predict(cluster_scaled)

pca = PCA(n_components=2, random_state=RANDOM_STATE)
pca_coords = pca.fit_transform(cluster_scaled)

cluster_result_df = df[["title", "type", "listed_in", "release_year", "rating", "primary_country"]].copy()
cluster_result_df["cluster"] = cluster_labels
cluster_result_df["pca_1"] = pca_coords[:, 0]
cluster_result_df["pca_2"] = pca_coords[:, 1]

cluster_interpretation: dict[int, dict] = {}
for c in sorted(cluster_result_df["cluster"].unique()):
    subset_idx = cluster_result_df["cluster"] == c
    stats = cluster_input[subset_idx].mean()
    top_genre = cluster_result_df.loc[subset_idx, "listed_in"].str.split(", ").explode().value_counts().idxmax()
    top_type = cluster_result_df.loc[subset_idx, "type"].value_counts().idxmax()
    cluster_interpretation[int(c)] = {
        "size": int(subset_idx.sum()),
        "avg_release_year": round(float(stats["release_year"]), 1),
        "avg_content_age": round(float(stats["content_age"]), 1),
        "avg_genres": round(float(stats["number_of_genres"]), 1),
        "avg_cast": round(float(stats["number_of_cast_members"]), 1),
        "dominant_type": top_type,
        "dominant_genre": top_genre,
    }

save(final_kmeans, "kmeans.joblib")
cluster_results_bundle = {
    "cluster_data": cluster_result_df,
    "elbow_data": {"k_range": k_list, "inertias": inertias, "silhouettes": silhouettes, "optimal_k": optimal_k},
    "cluster_interpretation": cluster_interpretation,
    "cluster_scaler": cluster_scaler,
    "pca_model": pca,
    "cluster_features": CLUSTER_FEATURES,
}
save(cluster_results_bundle, "cluster_results.joblib")

# ---------------------------------------------------------------------------
# 4. RECOMMENDATION SYSTEM — TF-IDF (no full similarity matrix persisted)
# ---------------------------------------------------------------------------
log("Building content-based recommendation engine ...")

df_rec = df.reset_index(drop=True).copy()
df_rec["combined_features"] = (
    df_rec["description"] + " " +
    df_rec["listed_in"] + " " +
    df_rec["cast"] + " " +
    df_rec["director"] + " " +
    df_rec["country"] + " " +
    df_rec["rating"]
)

tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
tfidf_matrix = tfidf_vectorizer.fit_transform(df_rec["combined_features"])

# NOTE: the full NxN cosine-similarity matrix is deliberately NOT persisted
# (8,807 x 8,807 floats would be ~600+ MB — impractical to ship). Only the
# compact sparse TF-IDF matrix (a few MB) is saved; the app computes
# similarity for just the queried row against all others on demand — a
# sub-second sparse matrix multiply, not a retrain.
title_index = pd.Series(df_rec.index, index=df_rec["title"].str.lower()).groupby(level=0).first()

save(tfidf_vectorizer, "tfidf_vectorizer.joblib")
save(tfidf_matrix, "tfidf_matrix.joblib")
save(title_index, "title_index.joblib")
df_rec[["title", "type", "listed_in", "rating", "country", "release_year", "description"]].to_pickle(
    os.path.join(MODELS_DIR, "titles_metadata.pkl")
)

# ---------------------------------------------------------------------------
# 4b. LIVE PREDICTION DEFAULTS — small artifact for the Live Prediction page
# ---------------------------------------------------------------------------
# The Live Prediction form deliberately does not collect a Genre input (genre
# is excluded from the classifier entirely to prevent leakage — see above),
# but the model still expects a `number_of_genres` value. We save the
# dataset-wide average here so inference time can fill it in sensibly rather
# than guessing 0, which would misrepresent a typical title.
log("Saving live-prediction inference defaults ...")
# Filter out 3 known malformed "rating" values in the raw dataset (e.g. "74 min")
# — a documented data-entry quirk where a handful of standup-special rows have
# their duration accidentally shifted into the rating column. These aren't real
# rating categories, so they're excluded from the dropdown shown to users.
valid_ratings = sorted(r for r in df["rating"].unique().tolist() if "min" not in str(r))
inference_defaults = {
    "reference_year": reference_year,
    "default_number_of_genres": float(df["number_of_genres"].mean()),
    "rating_options": valid_ratings,
    "min_release_year": int(df["release_year"].min()),
    "max_release_year": int(df["release_year"].max()),
}
save(inference_defaults, "inference_defaults.joblib")

# ---------------------------------------------------------------------------
# 5. INSIGHTS — pre-computed dynamic findings for the Insights page
# ---------------------------------------------------------------------------log("Computing dataset + model insights ...")
total = len(df)
movie_pct = (df["type"] == "Movie").mean()
added_after_2016_pct = (df["year_added"] >= 2016).mean()
top_country = df["country"].str.split(", ").explode().value_counts()
top_country = top_country[top_country.index != "Unknown"].idxmax()
top_genre = df["listed_in"].str.split(", ").explode().value_counts().idxmax()
best_silhouette = max(silhouettes)

insights = {
    "movie_pct": movie_pct,
    "added_after_2016_pct": added_after_2016_pct,
    "top_country": top_country,
    "top_genre": top_genre,
    "best_model_name": best_model_name,
    "best_model_f1": float(results_df.iloc[0]["F1 Score"]),
    "optimal_k": optimal_k,
    "best_silhouette": best_silhouette,
}
save(insights, "insights.joblib")

# ---------------------------------------------------------------------------
# 6. VERIFY
# ---------------------------------------------------------------------------
log("Verifying artifacts ...")
required = [
    "raw_data.pkl", "processed_data.pkl", "cleaning_report.joblib",
    "classification_model.joblib", "preprocessor.joblib", "preprocessor_template.joblib",
    "scaler.joblib", "metrics.joblib", "test_split.joblib",
    "kmeans.joblib", "cluster_results.joblib",
    "tfidf_vectorizer.joblib", "tfidf_matrix.joblib", "title_index.joblib", "titles_metadata.pkl",
    "inference_defaults.joblib", "insights.joblib", "model_metadata.joblib", "explanation_stats.joblib",
]
missing = [f for f in required if not os.path.exists(os.path.join(MODELS_DIR, f))]
if missing:
    raise FileNotFoundError(f"Missing artifacts after training: {missing}")

log("All artifacts saved successfully to models/. Training complete.")
print("\n" + results_df.drop(columns=["CV F1 Std"]).to_string(index=False))
