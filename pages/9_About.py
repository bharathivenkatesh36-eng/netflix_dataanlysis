"""Page 9 - Project documentation and methodology."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.loader import artifacts_exist, load_processed_data
from utils.styling import apply_netflix_theme, page_header

st.set_page_config(page_title="About | Netflix Dashboard", page_icon=":material/info:", layout="wide")
apply_netflix_theme()
page_header("About This Project", "Architecture, methodology, dataset context, and project scope", "info")

if artifacts_exist():
    df = load_processed_data()
    dataset_stats = [
        ("Titles", f"{len(df):,}"),
        ("Features", f"{df.shape[1]}"),
        ("Release coverage", f"{int(df['release_year'].min())}-{int(df['release_year'].max())}"),
        ("Content types", "Movie / TV Show"),
    ]
else:
    dataset_stats = [("Dataset", "Artifacts unavailable")]

st.markdown(
    """
    <section class="about-hero">
        <div class="hero-kicker">PROJECT BRIEF</div>
        <h1>Netflix Content Intelligence</h1>
        <p>An end-to-end data product that turns catalog metadata into analysis, model evaluation,
        recommendations, and explainable live predictions.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

st.markdown("### Dataset snapshot")
stat_cols = st.columns(len(dataset_stats))
for col, (label, value) in zip(stat_cols, dataset_stats):
    with col:
        st.markdown(
            f'<div class="insight-stat"><div class="insight-stat-label">{label}</div><div class="insight-stat-value">{value}</div></div>',
            unsafe_allow_html=True,
        )

left, right = st.columns([1.1, 0.9])
with left:
    st.markdown("### What this project answers")
    st.markdown(
        """
        Streaming catalogs contain useful signals, but raw metadata is difficult to compare at
        scale. This dashboard makes the catalog legible for four practical questions: what the
        library contains, how it has changed, whether content type can be predicted honestly,
        and which titles resemble a selected title without user history.
        """
    )
    st.markdown("### Design principles")
    principles = [
        ("Reproducible", "Saved artifacts keep the dashboard fast and make results repeatable."),
        ("Leakage-aware", "Features that reveal the target are excluded before classification."),
        ("Explainable", "Each chart and model result is paired with a plain-language takeaway."),
    ]
    for title, description in principles:
        st.markdown(
            f'<div class="about-principle"><strong>{title}</strong><span>{description}</span></div>',
            unsafe_allow_html=True,
        )

with right:
    st.markdown("### Technology stack")
    st.markdown(
        """
        <div class="netflix-card about-stack">
            <div><b>Application</b><span>Python, Streamlit</span></div>
            <div><b>Data</b><span>Pandas, NumPy</span></div>
            <div><b>Visualization</b><span>Plotly Express, Graph Objects</span></div>
            <div><b>Machine learning</b><span>scikit-learn, PCA, K-Means</span></div>
            <div><b>Persistence</b><span>Joblib, cached artifact loaders</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("### Workflow")
workflow = [
    ("01", "Dataset Analysis", "Inspect shape, quality, types, missingness, and uniqueness."),
    ("02", "Preprocessing", "Clean values, engineer features, and preserve meaningful missingness."),
    ("03", "EDA", "Explore catalog mix, ratings, geography, genres, people, and time."),
    ("04", "Supervised ML", "Compare five leakage-safe classifiers with cross-validation."),
    ("05", "Model Performance", "Review metrics, confusion matrices, ROC behavior, and importance."),
    ("06", "Unsupervised ML", "Use K-Means, elbow/silhouette analysis, and PCA to find segments."),
    ("07", "Recommendations", "Use TF-IDF and on-demand cosine similarity for related titles."),
    ("08", "Live Prediction", "Transform a new title and predict Movie versus TV Show."),
]
workflow_cols = st.columns(4)
for index, (number, title, description) in enumerate(workflow):
    with workflow_cols[index % 4]:
        st.markdown(
            f'<div class="workflow-card"><div class="workflow-number">{number}</div><h4>{title}</h4><p>{description}</p></div>',
            unsafe_allow_html=True,
        )

st.markdown("### Methodology notes")
method_cols = st.columns(3)
methodology = [
    ("Classification", "The pipeline uses a held-out test split and stratified cross-validation. Genre and duration-derived leakage features are excluded."),
    ("Recommendations", "Metadata fields are combined into text, transformed with TF-IDF, and compared with cosine similarity only for the requested title."),
    ("Artifacts", "Training happens offline in train_models.py. Pages load cached persisted models and reports rather than retraining during use."),
]
for col, (title, description) in zip(method_cols, methodology):
    with col:
        st.markdown(f'<div class="netflix-card methodology-card"><h4>{title}</h4><p>{description}</p></div>', unsafe_allow_html=True)

st.markdown("### Project structure")
st.code(
    """app.py                 # Streamlit entry point
train_models.py        # Offline training and artifact generation
data/                  # Source Netflix catalog
models/                # Saved models, metrics, and derived results
utils/                 # Loading, preprocessing, and shared styling
pages/                 # Analysis, ML, recommendation, and prediction views""",
    language="text",
)
