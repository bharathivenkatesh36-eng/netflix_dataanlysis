"""Page 7 — Content-Based Recommendation System."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.loader import artifacts_exist, load_titles_metadata, recommend_titles
from utils.styling import apply_netflix_theme, insight_box, page_header

st.set_page_config(page_title="Recommendations | Netflix Dashboard", page_icon=":material/auto_awesome:", layout="wide")
apply_netflix_theme()
page_header("Recommendation System", "Content-based recommendations using TF-IDF & Cosine Similarity", "auto_awesome")

if not artifacts_exist():
    st.error("Model artifacts not found. Run `python train_models.py` first.")
    st.stop()

with st.expander("How This Recommender Works", expanded=False):
    st.markdown(
        """
        This is a **content-based** recommender, not collaborative filtering — the dataset has
        no user watch history or ratings, so there's no way to build "users who liked X also
        liked Y". Instead, it recommends titles that are **textually and categorically similar**
        to a given title, based only on that title's own metadata.

        **Pipeline:**
        1. `description`, `listed_in`, `cast`, `director`, `country`, and `rating` are combined
           into one text blob per title.
        2. A `TfidfVectorizer` (max 5,000 features, English stop words removed) converts every
           title's blob into a weighted term vector — fit once in `train_models.py`.
        3. At request time, **cosine similarity** is computed between the query title's vector
           and every other title's vector, and the top matches are returned.

        **Memory design:** the full 8,807 × 8,807 similarity matrix is **not** stored (it
        would be ~600+ MB). Only the compact sparse TF-IDF matrix (a few MB) is saved, and
        similarity is computed for just the searched title against every other title on
        demand — a sub-second sparse matrix multiply, not a retrain.
        """
    )

meta = load_titles_metadata()
titles_list = sorted(meta["title"].unique().tolist())

c1, c2 = st.columns([2, 1])
with c1:
    selected_title = st.selectbox(
        "Search for a movie or TV show",
        options=titles_list,
        index=titles_list.index("Stranger Things") if "Stranger Things" in titles_list else 0,
    )
with c2:
    top_n = st.slider("Number of recommendations", 5, 20, 10)

if st.button("Get Recommendations", type="primary"):
    with st.spinner("Finding similar titles..."):
        results, error = recommend_titles(selected_title, n=top_n)

    if error:
        st.error(error)
    else:
        source_row = meta[meta["title"] == selected_title].iloc[0]
        st.markdown(
            f"""
            <div class="netflix-card">
                <span class="netflix-badge">{source_row['type']}</span>
                <span style="color:#B3B3B3; font-size:0.85rem;">{source_row['release_year']} · {source_row['rating']} · {source_row['country']}</span>
                <h4 style="margin-top:0.4rem;">{source_row['title']}</h4>
                <p style="color:#B3B3B3; font-size:0.9rem;">{source_row['description']}</p>
                <p style="color:#B3B3B3; font-size:0.85rem;"><i>Genres: {source_row['listed_in']}</i></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(f"### Top {len(results)} Titles Similar to *{selected_title}*")
        st.dataframe(
            results.style.background_gradient(subset=["Similarity Score"], cmap="Reds"),
            width="stretch", hide_index=True, height=min(45 * len(results) + 40, 560),
        )
        insight_box(
            "Similarity Score (0-1) reflects how much overlap exists between titles' "
            "descriptions, genres, cast, director, country, and rating — higher means more similar."
        )

        with st.expander("View as cards"):
            cols = st.columns(3)
            for i, (_, row) in enumerate(results.iterrows()):
                with cols[i % 3]:
                    st.markdown(
                        f"""
                        <div class="netflix-card" style="min-height:150px;">
                            <span class="netflix-badge">{row['Type']}</span>
                            <span style="color:#B3B3B3; font-size:0.78rem;">{row['Release Year']}</span>
                            <h4 style="margin-top:0.4rem; font-size:1rem;">{row['Title']}</h4>
                            <p style="color:#E50914; font-size:0.85rem; font-weight:700;">
                                {row['Similarity Score']:.1%} match
                            </p>
                            <p style="color:#B3B3B3; font-size:0.78rem;">{row['Genre']}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
else:
    st.info("Pick a title above and click **Get Recommendations** to see similar content.")
