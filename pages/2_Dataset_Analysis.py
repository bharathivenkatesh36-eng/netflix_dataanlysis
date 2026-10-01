"""Page 2 — Dataset Analysis."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.loader import artifacts_exist, load_raw_data
from utils.styling import apply_netflix_theme, insight_box, netflix_plotly_layout, page_header, render_kpi_cards

st.set_page_config(page_title="Dataset Analysis | Netflix Dashboard", page_icon=":material/database:", layout="wide")
apply_netflix_theme()
page_header("Dataset Analysis", "A first look at the raw netflix_titles dataset", "database")

if not artifacts_exist():
    st.error("Model artifacts not found. Run `python train_models.py` first.")
    st.stop()

df = load_raw_data()

COLUMN_DESCRIPTIONS = {
    "show_id": "Unique identifier for each title",
    "type": "Content type — Movie or TV Show",
    "title": "Name of the movie or TV show",
    "director": "Director(s) of the title (may be missing)",
    "cast": "Main cast members, comma-separated (may be missing)",
    "country": "Country or countries of production (may be missing)",
    "date_added": "Date the title was added to Netflix",
    "release_year": "Year the title was originally released",
    "rating": "Content/age rating (e.g. TV-MA, PG-13)",
    "duration": "Movie length in minutes, or number of TV seasons",
    "listed_in": "Genre(s) / category tags, comma-separated",
    "description": "Short synopsis of the title",
}

# ---------------------------------------------------------------------------
# Shape & KPIs
# ---------------------------------------------------------------------------
st.markdown("### Dataset Shape")
render_kpi_cards([
    {"label": "Rows", "value": f"{df.shape[0]:,}"},
    {"label": "Columns", "value": f"{df.shape[1]}"},
    {"label": "Missing Cells", "value": f"{int(df.isnull().sum().sum()):,}"},
    {"label": "Duplicate Rows", "value": f"{int(df.duplicated().sum()):,}"},
])
insight_box(
    f"The raw dataset has **{df.shape[0]:,} rows** (one per title) and **{df.shape[1]} columns** "
    "(metadata fields). Rows and columns are checked first in any analysis — they set the "
    "scale of everything that follows."
)

st.write("")
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview & Column Names", "Data Types & Info", "Missing Values", "Unique Values", "Statistical Summary"]
)

with tab1:
    c1, c2 = st.columns([1.4, 1])
    with c1:
        st.markdown("#### Sample Dataset")
        st.dataframe(df.head(10), width="stretch")
        insight_box("A quick preview confirms the columns look as expected and gives a feel for typical values before deeper analysis.")
    with c2:
        st.markdown("#### Column Names & Descriptions")
        desc_df = pd.DataFrame(
            {"Column": list(COLUMN_DESCRIPTIONS.keys()), "Description": list(COLUMN_DESCRIPTIONS.values())}
        )
        st.dataframe(desc_df, width="stretch", height=400, hide_index=True)

with tab2:
    dtypes_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Non-Null Count": df.notnull().sum().values,
        "Null Count": df.isnull().sum().values,
    })
    st.markdown("#### Dataset Information")
    st.dataframe(dtypes_df, width="stretch", hide_index=True)
    insight_box(
        "Most fields are stored as text (object) — including numeric-looking ones like "
        "`duration`, since it mixes minutes for movies with seasons for TV shows. "
        "`date_added` needs explicit datetime conversion before it's usable."
    )

with tab3:
    missing = df.isnull().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
    c1, c2 = st.columns([1, 1.3])
    with c1:
        st.markdown("#### Missing Values by Column")
        st.dataframe(missing_df, width="stretch")
    with c2:
        st.markdown("#### Missing Value Chart")
        missing_chart = missing_df.reset_index(names="Column")
        fig = px.bar(missing_chart, x="Missing %", y="Column", orientation="h", title="Missing Values by Column")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(netflix_plotly_layout(fig, height=380), width="stretch")
    insight_box(
        f"`{missing.index[0]}` has the most missing values "
        f"({missing_pct.iloc[0]}% of rows). director, cast, and country are the three "
        "fields most often left blank in the raw catalog data — all addressed on the "
        "Data Preprocessing page."
    )

with tab4:
    unique_df = pd.DataFrame({
        "Column": df.columns,
        "Unique Values": [df[c].nunique() for c in df.columns],
    }).sort_values("Unique Values", ascending=False)
    st.markdown("#### Unique Values per Column")
    st.dataframe(unique_df, width="stretch", hide_index=True)
    insight_box(
        "`title` is (almost) unique per row as expected. `type` and `rating` have very few "
        "distinct values, making them well suited to categorical encoding rather than text vectorization."
    )

with tab5:
    st.markdown("#### Statistical Summary — Numerical Columns")
    st.dataframe(df.describe().T, width="stretch")
    st.markdown("#### Statistical Summary — Categorical Columns")
    st.dataframe(df.describe(include=[object]).T, width="stretch")
    insight_box(
        "release_year spans several decades, but the statistical summary alone hides the "
        "sharp recent-year skew — see the Release Year Trend chart on the EDA page for the full picture."
    )
