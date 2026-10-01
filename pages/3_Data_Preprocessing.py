"""Page 3 — Data Preprocessing."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.loader import artifacts_exist, load_raw_data
from utils.preprocessing import (
    CLASSIFIER_CATEGORICAL_FEATURES,
    CLASSIFIER_EXCLUDED_FEATURES,
    CLASSIFIER_FEATURE_COLUMNS,
    CLASSIFIER_NUMERIC_FEATURES,
    CLASSIFIER_TEXT_FEATURE,
    clean_data,
    engineer_features,
)
from utils.styling import apply_netflix_theme, insight_box, page_header

st.set_page_config(page_title="Data Preprocessing | Netflix Dashboard", page_icon=":material/tune:", layout="wide")
apply_netflix_theme()
page_header("Data Preprocessing", "Every cleaning, engineering, and feature-selection step, explained", "tune")

if not artifacts_exist():
    st.error("Model artifacts not found. Run `python train_models.py` first.")
    st.stop()

st.markdown(
    """
    <div class="insight-box" style="margin-bottom:1.6rem;">
    This page runs the exact same <code>clean_data()</code> and <code>engineer_features()</code>
    functions from <code>utils/preprocessing.py</code> that <code>train_models.py</code> uses —
    so what you see here is guaranteed to match what the models were actually trained on. This
    is plain pandas cleaning, not model training, so it's cheap to (re)run on every page load.
    </div>
    """,
    unsafe_allow_html=True,
)

raw_df = load_raw_data()
cleaned_df, report = clean_data(raw_df)
processed_df = engineer_features(cleaned_df)

# ---------------------------------------------------------------------------
# Step 1 — Duplicates
# ---------------------------------------------------------------------------
st.markdown("### Step 1 · Remove Duplicates")
c1, c2, c3 = st.columns(3)
c1.metric("Rows Before", f"{report['raw_shape'][0]:,}")
c2.metric("Duplicates Removed", f"{report['duplicates_removed']:,}")
c3.metric("Rows After", f"{report['cleaned_shape'][0]:,}")
st.caption("`df.drop_duplicates()` — duplicate rows add no new information and would silently bias counts in the EDA and class balance in the ML models.")

st.markdown("---")

# ---------------------------------------------------------------------------
# Step 2 — Missing values
# ---------------------------------------------------------------------------
st.markdown("### Step 2 · Handle Missing Values")
st.markdown(
    """
    - **`director`, `cast`, `country`** → filled with `"Unknown"` — missing is itself meaningful
      here (many titles genuinely have no listed director), so it's kept explicit rather than
      dropping rows.
    - **`rating`** → filled with the **mode** (most frequent rating).
    - **`description`, `duration`** → filled with an empty string / `"Unknown"` so downstream
      text-vectorization and parsing steps don't break.
    """
)
before_after = pd.DataFrame({
    "Column": ["director", "cast", "country", "rating", "description"],
    "Missing Before": [report["missing_before"].get(c, 0) for c in ["director", "cast", "country", "rating", "description"]],
    "Missing After": [report["missing_after"].get(c, 0) for c in ["director", "cast", "country", "rating", "description"]],
})
st.dataframe(before_after, width="stretch", hide_index=True)
insight_box("Every targeted column goes from non-trivial missing counts to exactly zero.")

st.markdown("---")

# ---------------------------------------------------------------------------
# Step 3 — Date conversion
# ---------------------------------------------------------------------------
st.markdown("### Step 3 · Date Conversion")
st.markdown(
    """
    `date_added` arrives as free-text strings with inconsistent whitespace. It's stripped and
    parsed with `pd.to_datetime(..., errors="coerce")`, turning any unparsable values into
    `NaT` instead of crashing the pipeline. `year_added` and `month_added` are then derived
    from this clean datetime column.
    """
)
st.dataframe(processed_df[["date_added", "year_added", "month_added"]].head(5), width="stretch", hide_index=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# Step 4 — Feature engineering
# ---------------------------------------------------------------------------
st.markdown("### Step 4 · Feature Engineering")
st.markdown(
    """
    | New Feature | How it's built | Why |
    |---|---|---|
    | `content_age` | `newest_release_year - release_year` | How old a title was at release, relative to the newest release year in the catalog |
    | `duration_minutes` | Parsed from `duration` for Movies only | Numeric movie length |
    | `number_of_seasons` | Parsed from `duration` for TV Shows only | Numeric season count |
    | `number_of_countries` / `number_of_genres` / `number_of_cast_members` / `number_of_directors` | Count of comma-separated items | Captures scale of co-production, genre-tagging, and cast/crew size as a single number |
    | `primary_country` / `primary_genre` | First listed value | A clean single-label categorical version of a multi-value field |
    """
)
st.dataframe(
    processed_df[[
        "title", "content_age", "duration_minutes", "number_of_seasons",
        "number_of_countries", "number_of_genres", "primary_country", "primary_genre",
    ]].head(6),
    width="stretch", hide_index=True,
)

st.markdown("---")

# ---------------------------------------------------------------------------
# Step 5 — Encoding
# ---------------------------------------------------------------------------
st.markdown("### Step 5 · Encoding")
st.markdown(
    """
    - **`rating`, `primary_country`** → **One-Hot Encoding** (`OneHotEncoder(handle_unknown="ignore")`)
      inside the modeling pipeline.
    - **`description`** → **TF-IDF vectorization** (200 features for the classifier; 5,000 for the recommender).
    - **`type`** (the supervised learning target) → binary label encoding (`Movie` = 0, `TV Show` = 1).

    Encoding is fit **only on the training split**, inside a scikit-learn `Pipeline`, so there's
    no leakage from the test set into the encoders.
    """
)

st.markdown("---")

# ---------------------------------------------------------------------------
# Step 6 — Scaling
# ---------------------------------------------------------------------------
st.markdown("### Step 6 · Scaling")
st.markdown(
    """
    All numeric features are standardized with `StandardScaler` (zero mean, unit variance)
    before being fed to distance/gradient-based models (KNN, SVM, Logistic Regression) and
    before K-Means clustering — without scaling, `release_year` would dominate purely because
    of its larger numeric range.
    """
)

st.markdown("---")

# ---------------------------------------------------------------------------
# Step 7 — Feature selection
# ---------------------------------------------------------------------------
st.markdown("### Step 7 · Feature Selection (Leakage Prevention)")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Included in the classifier**")
    st.markdown("\n".join([f"- `{f}`" for f in CLASSIFIER_FEATURE_COLUMNS]))
with c2:
    st.markdown("**Excluded (leakage risk)**")
    st.markdown("\n".join([f"- `{f}`" for f in CLASSIFIER_EXCLUDED_FEATURES]))
st.warning(
    "**Why `listed_in` is excluded:** Netflix's own genre tags spell out \"TV Shows\" / "
    "\"TV Dramas\" for ~96% of TV Show rows and never for Movies — including it as a TF-IDF "
    "feature was verified to push every classifier to ~100% accuracy for the wrong reason. "
    "`duration` / `duration_minutes` / `number_of_seasons` are excluded for the same reason: "
    "each is populated for only one class, so its presence/absence alone gives the label away. "
    "See the Supervised ML page for the honest, leakage-safe results."
)

st.markdown("---")

# ---------------------------------------------------------------------------
# Final preview
# ---------------------------------------------------------------------------
st.markdown("### Final Processed Dataset")
d1, d2 = st.columns(2)
d1.metric("Raw shape", f"{report['raw_shape'][0]:,} × {report['raw_shape'][1]}")
d2.metric("Processed shape", f"{processed_df.shape[0]:,} × {processed_df.shape[1]}")
st.dataframe(processed_df.head(10), width="stretch")

with st.expander("Show all processed column names"):
    st.write(list(processed_df.columns))
