"""
utils/preprocessing.py
=======================
Pure, reusable data-cleaning and feature-engineering functions.

These functions contain NO machine-learning model code and are cheap to run
(plain pandas on ~8,800 rows), so they are safely called from BOTH:
    1. `train_models.py` (to build the dataset the models are trained on), and
    2. `pages/3_Data_Preprocessing.py` (to show a live before/after demo).

Sharing this module means the app's "how the data was cleaned" explanation is
always guaranteed to match what the models were actually trained on — there is
only one implementation, not two that could drift apart. This is distinct
from *model* training/prediction, which only ever happens in
`train_models.py` and is never repeated inside Streamlit.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Remove duplicates and handle missing values.

    Returns the cleaned dataframe and a report dict describing what changed,
    so the UI can show an honest before/after comparison.
    """
    missing_before = df.isnull().sum()
    duplicates_found = int(df.duplicated().sum())

    cleaned = df.drop_duplicates().copy()

    cleaned["director"] = cleaned["director"].fillna("Unknown")
    cleaned["cast"] = cleaned["cast"].fillna("Unknown")
    cleaned["country"] = cleaned["country"].fillna("Unknown")
    cleaned["rating"] = cleaned["rating"].fillna(cleaned["rating"].mode()[0])
    cleaned["description"] = cleaned["description"].fillna("")
    cleaned["duration"] = cleaned["duration"].fillna("Unknown")

    cleaned["date_added"] = cleaned["date_added"].astype(str).str.strip()
    cleaned["date_added"] = pd.to_datetime(cleaned["date_added"], errors="coerce")

    missing_after = cleaned.isnull().sum()

    report = {
        "raw_shape": df.shape,
        "cleaned_shape": cleaned.shape,
        "duplicates_removed": duplicates_found,
        "missing_before": missing_before.to_dict(),
        "missing_after": missing_after.to_dict(),
    }
    return cleaned, report


def _parse_duration_minutes(row: pd.Series) -> float:
    """Extract movie runtime in minutes. Returns NaN for TV shows."""
    if row["type"] == "Movie" and isinstance(row["duration"], str) and "min" in row["duration"]:
        match = re.search(r"(\d+)", row["duration"])
        return float(match.group(1)) if match else np.nan
    return np.nan


def _parse_number_of_seasons(row: pd.Series) -> float:
    """Extract TV show season count. Returns NaN for movies."""
    if row["type"] == "TV Show" and isinstance(row["duration"], str) and "Season" in row["duration"]:
        match = re.search(r"(\d+)", row["duration"])
        return float(match.group(1)) if match else np.nan
    return np.nan


def count_items(text: Any) -> int:
    """Count comma-separated items in a text field (0 for empty/Unknown/NaN).

    Public because it's also used at inference time (see
    `build_live_input_features` below) to engineer a single new record with
    the exact same counting logic used during training — not a re-derivation
    that could quietly drift from it.
    """
    if not isinstance(text, str) or text.strip() == "" or text == "Unknown":
        return 0
    return len([item for item in text.split(",") if item.strip()])


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add every engineered column used downstream (EDA, ML, clustering).

    Expects `df` to already be cleaned (see `clean_data`).
    """
    out = df.copy()

    out["year_added"] = out["date_added"].dt.year
    out["month_added"] = out["date_added"].dt.month_name()

    reference_year = int(out["release_year"].max())
    out["content_age"] = reference_year - out["release_year"]

    out["duration_minutes"] = out.apply(_parse_duration_minutes, axis=1)
    out["number_of_seasons"] = out.apply(_parse_number_of_seasons, axis=1)

    out["number_of_countries"] = out["country"].apply(count_items)
    out["number_of_genres"] = out["listed_in"].apply(count_items)
    out["number_of_cast_members"] = out["cast"].apply(count_items)
    out["number_of_directors"] = out["director"].apply(count_items)

    out["primary_country"] = out["country"].apply(
        lambda x: x.split(",")[0].strip() if isinstance(x, str) else "Unknown"
    )
    out["primary_genre"] = out["listed_in"].apply(
        lambda x: x.split(",")[0].strip() if isinstance(x, str) else "Unknown"
    )
    return out


# ---------------------------------------------------------------------------
# Feature selection — the leakage-safe classifier feature set
# ---------------------------------------------------------------------------
# `listed_in` (genre) is EXCLUDED here on purpose: Netflix's own genre tags
# spell out "TV Shows" / "TV Dramas" / etc. for ~96% of TV Show rows and never
# for Movies, so vectorizing it as a feature lets the target leak into the
# input almost verbatim (verified: it drives every classifier to ~100%
# accuracy for the wrong reason). `duration` / `duration_minutes` /
# `number_of_seasons` are excluded for the same reason — each is populated
# for only one class, so its mere presence/absence gives the label away.
CLASSIFIER_NUMERIC_FEATURES: list[str] = [
    "release_year", "content_age", "number_of_countries",
    "number_of_genres", "number_of_cast_members", "number_of_directors",
]
CLASSIFIER_CATEGORICAL_FEATURES: list[str] = ["rating", "primary_country"]
CLASSIFIER_TEXT_FEATURE: str = "description"
CLASSIFIER_FEATURE_COLUMNS: list[str] = (
    CLASSIFIER_NUMERIC_FEATURES + CLASSIFIER_CATEGORICAL_FEATURES + [CLASSIFIER_TEXT_FEATURE]
)
CLASSIFIER_EXCLUDED_FEATURES: list[str] = ["listed_in", "duration", "duration_minutes", "number_of_seasons"]

# Numeric features used for K-Means clustering (unsupervised learning)
CLUSTER_FEATURES: list[str] = [
    "release_year", "content_age", "number_of_countries",
    "number_of_genres", "number_of_cast_members", "number_of_directors",
]


def build_live_input_features(
    release_year: int,
    rating: str,
    country: str,
    director: str,
    cast: str,
    description: str,
    reference_year: int,
    default_number_of_genres: float,
) -> pd.DataFrame:
    """Engineer a single-row feature dataframe for the Live Prediction page.

    Uses the exact same field-counting logic (`count_items`) as
    `engineer_features()` above, so a brand-new title is turned into features
    the saved model actually recognizes — not a parallel, potentially
    inconsistent re-implementation.

    Note on `number_of_genres`: the Live Prediction form deliberately does
    NOT collect a Genre input, for the same reason `listed_in` is excluded
    from the classifier entirely (see `CLASSIFIER_EXCLUDED_FEATURES` above) —
    Netflix's genre tags leak the target label. Since the model still expects
    a `number_of_genres` value as one of its numeric features, it's filled
    with the dataset-wide average (computed once at training time and passed
    in here as `default_number_of_genres`) rather than an arbitrary 0, which
    would misrepresent a typical title and skew the prediction.
    """
    country = country.strip() if country and country.strip() else "Unknown"
    director = director.strip() if director and director.strip() else "Unknown"
    cast = cast.strip() if cast and cast.strip() else "Unknown"
    description = description.strip() if description else ""

    primary_country = country.split(",")[0].strip() if country != "Unknown" else "Unknown"

    row = {
        "release_year": release_year,
        "content_age": reference_year - release_year,
        "number_of_countries": count_items(country),
        "number_of_genres": default_number_of_genres,
        "number_of_cast_members": count_items(cast),
        "number_of_directors": count_items(director),
        "rating": rating,
        "primary_country": primary_country,
        "description": description,
    }
    return pd.DataFrame([row])[CLASSIFIER_FEATURE_COLUMNS]
