"""Page 4 — Exploratory Data Analysis."""

from __future__ import annotations

import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.helper import month_order, top_n_from_column
from utils.loader import artifacts_exist, load_processed_data
from utils.styling import apply_netflix_theme, insight_box, netflix_plotly_layout, page_header

st.set_page_config(page_title="EDA | Netflix Dashboard", page_icon=":material/bar_chart:", layout="wide")
apply_netflix_theme()
page_header("Exploratory Data Analysis", "Interactive charts across content type, genre, geography, and time", "bar_chart")

if not artifacts_exist():
    st.error("Model artifacts not found. Run `python train_models.py` first.")
    st.stop()

df = load_processed_data()

# ---------------------------------------------------------------------------
# Filters — Country, Genre, Rating, Release Year, Type
# ---------------------------------------------------------------------------
with st.expander("Filters", expanded=True):
    f1, f2, f3 = st.columns(3)
    with f1:
        type_filter = st.multiselect("Type", options=sorted(df["type"].unique()), default=list(df["type"].unique()))
    with f2:
        top_countries = df["primary_country"].value_counts().head(20).index.tolist()
        country_filter = st.multiselect("Country (top 20 by primary country)", options=top_countries, default=[])
    with f3:
        rating_filter = st.multiselect("Rating", options=sorted(df["rating"].unique()), default=[])

    f4, f5 = st.columns(2)
    with f4:
        top_genres = df["listed_in"].str.split(", ").explode().value_counts().head(25).index.tolist()
        genre_filter = st.multiselect("Genre (top 25)", options=top_genres, default=[])
    with f5:
        min_year, max_year = int(df["release_year"].min()), int(df["release_year"].max())
        year_range = st.slider("Release Year", min_year, max_year, (2000, max_year))

filtered = df[
    df["type"].isin(type_filter)
    & df["release_year"].between(year_range[0], year_range[1])
]
if country_filter:
    filtered = filtered[filtered["primary_country"].isin(country_filter)]
if rating_filter:
    filtered = filtered[filtered["rating"].isin(rating_filter)]
if genre_filter:
    filtered = filtered[filtered["listed_in"].apply(lambda x: any(g in x for g in genre_filter))]

st.caption(f"Showing **{len(filtered):,}** of {len(df):,} titles based on current filters.")

if filtered.empty:
    st.warning("No titles match the current filters. Try widening your selection.")
    st.stop()

movies_f = filtered[filtered["type"] == "Movie"]
tv_f = filtered[filtered["type"] == "TV Show"]

tabs = st.tabs([
    "Content Mix", "Genres & Ratings", "Geography", "Time Trends",
    "Duration", "People", "Correlation & Missingness",
])

# ---------------------------------------------------------------------------
# Tab 1 — Content mix
# ---------------------------------------------------------------------------
with tabs[0]:
    c1, c2 = st.columns(2)
    with c1:
        fig = px.pie(filtered, names="type", hole=0.45, title="Movies vs TV Shows")
        st.plotly_chart(netflix_plotly_layout(fig), width="stretch")
        insight_box(
            f"Movies make up {len(movies_f)/len(filtered):.0%} of the filtered catalog versus "
            f"{len(tv_f)/len(filtered):.0%} for TV Shows. Business takeaway: content strategy "
            "still leans heavily toward films over series."
        )
    with c2:
        fig = px.histogram(
            filtered, x="release_year", color="type", nbins=40, barmode="overlay",
            opacity=0.75, title="Release Year Trend by Type",
        )
        st.plotly_chart(netflix_plotly_layout(fig), width="stretch")
        insight_box("Both content types skew heavily toward recent release years, reflecting rapid catalog growth in the last decade.")

# ---------------------------------------------------------------------------
# Tab 2 — Genres & ratings
# ---------------------------------------------------------------------------
with tabs[1]:
    c1, c2 = st.columns(2)
    with c1:
        genre_counts = top_n_from_column(filtered, "listed_in", n=15)
        fig = px.bar(
            genre_counts, x="count", y="listed_in", orientation="h",
            title="Top 15 Genres", labels={"count": "Number of Titles", "listed_in": "Genre"},
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(netflix_plotly_layout(fig, height=520), width="stretch")
        insight_box(f"'{genre_counts.iloc[0]['listed_in']}' is the single most common genre tag — a strong signal for content acquisition priorities.")
    with c2:
        rating_counts = filtered["rating"].value_counts().reset_index()
        rating_counts.columns = ["rating", "count"]
        fig = px.bar(
            rating_counts, x="count", y="rating", orientation="h",
            title="Rating Distribution", labels={"count": "Number of Titles", "rating": "Rating"},
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(netflix_plotly_layout(fig, height=520), width="stretch")
        insight_box("Mature-audience ratings (TV-MA, TV-14) dominate the catalog, consistent with an adult-skewing content library.")

    genre_type = filtered.assign(genre=filtered["listed_in"].str.split(", ")).explode("genre")
    top_genre_list = genre_type["genre"].value_counts().head(10).index
    genre_type_top = genre_type[genre_type["genre"].isin(top_genre_list)]
    fig = px.treemap(genre_type_top, path=["type", "genre"], title="Top Genres by Content Type (Treemap)")
    st.plotly_chart(netflix_plotly_layout(fig, height=480), width="stretch")
    insight_box("Larger tiles mean more titles — this shows how the top genres split between Movies and TV Shows at a glance.")

# ---------------------------------------------------------------------------
# Tab 3 — Geography
# ---------------------------------------------------------------------------
with tabs[2]:
    country_counts = top_n_from_column(filtered, "country", n=15)
    fig = px.bar(
        country_counts, x="count", y="country", orientation="h",
        title="Top 15 Content-Producing Countries", labels={"count": "Number of Titles", "country": "Country"},
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(netflix_plotly_layout(fig, height=520), width="stretch")
    insight_box(f"'{country_counts.iloc[0]['country']}' leads content production in the current selection — a candidate market for continued investment.")

    world_counts = filtered["country"].str.split(", ").explode().str.strip().value_counts()
    world_counts = world_counts[world_counts.index != "Unknown"].reset_index()
    world_counts.columns = ["country", "count"]
    fig = px.choropleth(
        world_counts, locations="country", locationmode="country names", color="count",
        hover_name="country", color_continuous_scale="Reds", title="Netflix Titles by Country (World Map)",
    )
    st.plotly_chart(netflix_plotly_layout(fig, height=520), width="stretch")
    insight_box("Darker shades indicate more titles produced (fully or partly) in that country — highlighting geographic concentration of supply.")

# ---------------------------------------------------------------------------
# Tab 4 — Time trends
# ---------------------------------------------------------------------------
with tabs[3]:
    c1, c2 = st.columns(2)
    with c1:
        yearly = filtered.dropna(subset=["year_added"]).groupby(["year_added", "type"]).size().reset_index(name="count")
        fig = px.line(
            yearly, x="year_added", y="count", color="type", markers=True,
            title="Content Added by Year", labels={"year_added": "Year Added", "count": "Titles Added"},
        )
        st.plotly_chart(netflix_plotly_layout(fig), width="stretch")
        pct_after_2016 = (filtered["year_added"] >= 2016).mean()
        insight_box(f"{pct_after_2016:.0%} of titles in this selection were added from 2016 onward — additions accelerated sharply and peaked around 2019-2020.")
    with c2:
        monthly = filtered["month_added"].value_counts().reindex(month_order()).reset_index()
        monthly.columns = ["month", "count"]
        fig = px.bar(monthly, x="month", y="count", title="Monthly Additions (All Years)")
        st.plotly_chart(netflix_plotly_layout(fig), width="stretch")
        insight_box("Content additions are fairly evenly spread across months, with a slight uptick around year-end — useful for planning release cadence.")

# ---------------------------------------------------------------------------
# Tab 5 — Duration
# ---------------------------------------------------------------------------
with tabs[4]:
    c1, c2 = st.columns(2)
    with c1:
        if movies_f["duration_minutes"].notna().any():
            fig = px.histogram(
                movies_f, x="duration_minutes", nbins=30, title="Movie Duration Distribution",
                labels={"duration_minutes": "Duration (minutes)"},
            )
            st.plotly_chart(netflix_plotly_layout(fig), width="stretch")
            avg_dur = movies_f["duration_minutes"].mean()
            insight_box(f"The average movie runs about {avg_dur:.0f} minutes, with most titles clustered between 80-120 minutes.")
        else:
            st.info("No movies in the current filter selection.")
    with c2:
        if tv_f["number_of_seasons"].notna().any():
            season_counts = tv_f["number_of_seasons"].dropna().astype(int).value_counts().sort_index().reset_index()
            season_counts.columns = ["seasons", "count"]
            fig = px.bar(season_counts, x="seasons", y="count", title="TV Show Season Distribution")
            st.plotly_chart(netflix_plotly_layout(fig), width="stretch")
            pct_single = (tv_f["number_of_seasons"] == 1).mean()
            insight_box(f"{pct_single:.0%} of TV shows in this selection have only a single season — most series are not renewed multiple times.")
        else:
            st.info("No TV shows in the current filter selection.")

    if movies_f["duration_minutes"].notna().any():
        fig = px.box(
            movies_f, x="rating", y="duration_minutes", title="Movie Duration by Rating",
            labels={"duration_minutes": "Duration (minutes)"},
        )
        st.plotly_chart(netflix_plotly_layout(fig, height=480), width="stretch")
        insight_box("Box plots show both the typical runtime and the spread/outliers for each rating category.")

# ---------------------------------------------------------------------------
# Tab 6 — People
# ---------------------------------------------------------------------------
with tabs[5]:
    c1, c2 = st.columns(2)
    with c1:
        top_directors = top_n_from_column(filtered[filtered["director"] != "Unknown"], "director", n=10)
        fig = px.bar(
            top_directors, x="count", y="director", orientation="h", title="Top 10 Directors",
            labels={"count": "Number of Titles", "director": "Director"},
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(netflix_plotly_layout(fig, height=460), width="stretch")
        if not top_directors.empty:
            insight_box(f"{top_directors.iloc[0]['director']} has the most credited titles in this selection.")
    with c2:
        top_actors = top_n_from_column(filtered[filtered["cast"] != "Unknown"], "cast", n=10)
        fig = px.bar(
            top_actors, x="count", y="cast", orientation="h", title="Top 10 Actors",
            labels={"count": "Number of Titles", "cast": "Actor"},
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(netflix_plotly_layout(fig, height=460), width="stretch")
        if not top_actors.empty:
            insight_box(f"{top_actors.iloc[0]['cast']} appears in more titles than any other cast member in this selection.")

# ---------------------------------------------------------------------------
# Tab 7 — Correlation & Missingness
# ---------------------------------------------------------------------------
with tabs[6]:
    c1, c2 = st.columns(2)
    with c1:
        numeric_cols = [
            "release_year", "content_age", "number_of_countries",
            "number_of_genres", "number_of_cast_members", "number_of_directors",
        ]
        corr = filtered[numeric_cols].corr()
        fig = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            title="Correlation Heatmap (Engineered Numeric Features)", aspect="auto",
        )
        st.plotly_chart(netflix_plotly_layout(fig, height=480), width="stretch")
        insight_box("Most engineered numeric features are only weakly correlated with one another — a good sign for supervised models, since heavily correlated features add redundancy rather than signal.")
    with c2:
        st.markdown("##### Missing Value Heatmap")
        missing_mask = filtered.isnull().astype(int)
        # Sample rows for a readable heatmap on large datasets
        sample = missing_mask.sample(min(200, len(missing_mask)), random_state=42) if len(missing_mask) > 200 else missing_mask
        fig = px.imshow(
            sample.T, color_continuous_scale=["#1F1F1F", "#E50914"], aspect="auto",
            labels={"x": "Row (sample)", "y": "Column", "color": "Missing"},
        )
        fig.update_xaxes(showticklabels=False)
        st.plotly_chart(netflix_plotly_layout(fig, title="Missing Value Heatmap (sampled rows)", height=480), width="stretch")
        insight_box("Red cells mark missing values. duration_minutes and number_of_seasons are 'missing' by design — each is only populated for its matching content type.")
