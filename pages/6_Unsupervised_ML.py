"""Page 6 — Unsupervised Machine Learning."""

from __future__ import annotations

import sys
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.loader import artifacts_exist, load_cluster_results, load_kmeans_model
from utils.styling import apply_netflix_theme, insight_box, netflix_plotly_layout, page_header, render_kpi_cards

st.set_page_config(page_title="Unsupervised ML | Netflix Dashboard", page_icon=":material/hub:", layout="wide")
apply_netflix_theme()
page_header("Unsupervised Machine Learning", "Discovering natural content segments with K-Means clustering", "hub")

if not artifacts_exist():
    st.error("Model artifacts not found. Run `python train_models.py` first.")
    st.stop()

kmeans = load_kmeans_model()
bundle = load_cluster_results()
cluster_df = bundle["cluster_data"]
elbow_data = bundle["elbow_data"]
interpretation = bundle["cluster_interpretation"]
optimal_k = elbow_data["optimal_k"]

with st.expander("How This Clustering Was Built", expanded=False):
    st.markdown(
        f"""
        **Algorithm:** K-Means, fit on standardized numeric features:
        `{'`, `'.join(bundle['cluster_features'])}`.

        **Choosing k:** Both the **Elbow Method** (inertia vs. k) and **Silhouette Score**
        were computed for k = 2 to 10. k = 2 technically maximizes silhouette here, but it
        collapses into one huge "everything" cluster and one tiny outlier group — not useful
        to interpret. The best k in the presentable k = 3 to 8 range was chosen instead,
        giving **k = {optimal_k}**.

        **Visualization:** The scaled feature space is projected down to 2 dimensions with
        **PCA** purely for plotting — clustering itself happens in the full feature space,
        not on the 2D projection.

        *Hierarchical clustering and DBSCAN were considered as optional alternatives; K-Means
        was chosen as the primary method for its speed and straightforward interpretability at
        this dataset size — see the About page's Future Improvements for extending this.*
        """
    )

render_kpi_cards([
    {"label": "Optimal k", "value": str(optimal_k)},
    {"label": "Titles Clustered", "value": f"{len(cluster_df):,}"},
    {"label": "Best Silhouette Score", "value": f"{max(elbow_data['silhouettes']):.3f}"},
])

st.markdown("---")

# ---------------------------------------------------------------------------
# Elbow + Silhouette
# ---------------------------------------------------------------------------
c1, c2 = st.columns(2)
with c1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=elbow_data["k_range"], y=elbow_data["inertias"], mode="lines+markers",
        line=dict(color="#E50914", width=3), marker=dict(size=9),
    ))
    fig.add_vline(x=optimal_k, line_dash="dash", line_color="#B3B3B3",
                  annotation_text=f"chosen k={optimal_k}", annotation_position="top")
    fig.update_layout(xaxis_title="Number of Clusters (k)", yaxis_title="Inertia (WCSS)")
    st.plotly_chart(netflix_plotly_layout(fig, title="Elbow Method"), width="stretch")
    insight_box("Inertia (within-cluster sum of squares) keeps decreasing as k grows — the 'elbow' is where the rate of decrease visibly flattens.")
with c2:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=elbow_data["k_range"], y=elbow_data["silhouettes"], mode="lines+markers",
        line=dict(color="#F5576C", width=3), marker=dict(size=9),
    ))
    fig.add_vline(x=optimal_k, line_dash="dash", line_color="#B3B3B3",
                  annotation_text=f"chosen k={optimal_k}", annotation_position="top")
    fig.update_layout(xaxis_title="Number of Clusters (k)", yaxis_title="Silhouette Score")
    st.plotly_chart(netflix_plotly_layout(fig, title="Silhouette Score"), width="stretch")
    insight_box("Higher silhouette scores mean tighter, better-separated clusters. Scores here are modest (~0.24-0.28), which makes sense — content metadata forms overlapping, fuzzy groups rather than sharply separated ones.")

st.markdown("---")

# ---------------------------------------------------------------------------
# PCA cluster visualization
# ---------------------------------------------------------------------------
st.markdown("### 🗺️ Cluster Visualization (PCA-Reduced)")
fig = px.scatter(
    cluster_df, x="pca_1", y="pca_2", color=cluster_df["cluster"].astype(str),
    hover_data=["title", "type", "release_year"],
    title=f"K-Means Clusters (k={optimal_k}) — PCA Projection",
    labels={"color": "Cluster", "pca_1": "Principal Component 1", "pca_2": "Principal Component 2"},
)
fig.update_traces(marker=dict(size=6, opacity=0.6))
st.plotly_chart(netflix_plotly_layout(fig, height=560), width="stretch")
insight_box("Each point is one title, projected from 6 dimensions down to 2 for visualization. Color shows its assigned cluster.")

st.markdown("---")

# ---------------------------------------------------------------------------
# Cluster distribution & interpretation
# ---------------------------------------------------------------------------
c1, c2 = st.columns([1, 1.4])
with c1:
    st.markdown("### Cluster Distribution")
    dist = cluster_df["cluster"].value_counts().sort_index().reset_index()
    dist.columns = ["cluster", "count"]
    dist["cluster"] = dist["cluster"].astype(str)
    fig = px.bar(dist, x="cluster", y="count", title="Titles per Cluster")
    st.plotly_chart(netflix_plotly_layout(fig), width="stretch")

with c2:
    st.markdown("### Cluster Interpretation")
    for cid, info in interpretation.items():
        st.markdown(
            f"""
            <div class="netflix-card">
                <h4>Cluster {cid} — {info['size']:,} titles</h4>
                <p style="color:#B3B3B3; margin-bottom:0.2rem;">
                    Dominant type: <b style="color:white;">{info['dominant_type']}</b> &nbsp;•&nbsp;
                    Dominant genre: <b style="color:white;">{info['dominant_genre']}</b>
                </p>
                <p style="color:#B3B3B3; font-size:0.87rem;">
                    Avg. release year: <b style="color:white;">{info['avg_release_year']}</b> &nbsp;•&nbsp;
                    Avg. content age: <b style="color:white;">{info['avg_content_age']} yrs</b> &nbsp;•&nbsp;
                    Avg. genres/title: <b style="color:white;">{info['avg_genres']}</b> &nbsp;•&nbsp;
                    Avg. cast size: <b style="color:white;">{info['avg_cast']}</b>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("### Explore a Cluster")
chosen_cluster = st.selectbox("Select a cluster to view sample titles", options=sorted(cluster_df["cluster"].unique()))
sample = cluster_df[cluster_df["cluster"] == chosen_cluster][["title", "type", "listed_in", "release_year", "rating", "primary_country"]].head(15)
st.dataframe(sample, width="stretch", hide_index=True)
