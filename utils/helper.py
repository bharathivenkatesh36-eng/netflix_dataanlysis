"""
utils/helper.py
================
Small, reusable, dependency-light helper functions shared across pages.
Nothing here touches models or Streamlit widgets directly — pure functions
only, so they're easy to unit-test.
"""

from __future__ import annotations

import io
import textwrap
from datetime import datetime

import pandas as pd


def explode_column(df: pd.DataFrame, column: str, sep: str = ", ") -> pd.Series:
    """Split a comma-separated text column into one row per item and count
    occurrences. Used for genre / country / cast / director frequency counts.
    """
    return df[column].dropna().str.split(sep).explode().str.strip().value_counts()


def top_n_from_column(df: pd.DataFrame, column: str, n: int = 10, exclude: tuple[str, ...] = ("Unknown",)) -> pd.DataFrame:
    """Return the top-n most frequent comma-separated values in `column` as a
    tidy two-column dataframe: [column-name-singular, count].
    """
    counts = explode_column(df, column)
    counts = counts[~counts.index.isin(exclude)].head(n)
    result = counts.reset_index()
    result.columns = [column, "count"]
    return result


def format_seconds(seconds: float) -> str:
    """Human-readable duration for training/prediction time displays."""
    if seconds < 1:
        return f"{seconds * 1000:.1f} ms"
    if seconds < 60:
        return f"{seconds:.2f} s"
    minutes, secs = divmod(seconds, 60)
    return f"{int(minutes)}m {secs:.1f}s"


def format_count(n: int) -> str:
    """Thousands-separated integer for display in KPI cards."""
    return f"{n:,}"


def percentage_of(part: int, whole: int) -> str:
    """Safe percentage formatting; returns 'N/A' if whole is zero."""
    if not whole:
        return "N/A"
    return f"{part / whole:.1%}"


def safe_mode(series: pd.Series, default: str = "Unknown") -> str:
    """Most frequent value in a series, or `default` if the series is empty."""
    modes = series.mode()
    return modes.iloc[0] if not modes.empty else default


def month_order() -> list[str]:
    """Calendar month names in order, for consistent chart x-axis ordering."""
    return [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]


def truncate_text(text: str, max_len: int = 160) -> str:
    """Truncate long text (e.g. descriptions) for compact card display."""
    if not isinstance(text, str):
        return ""
    return text if len(text) <= max_len else text[: max_len - 1].rstrip() + "…"


# ---------------------------------------------------------------------------
# Live Prediction — input validation
# ---------------------------------------------------------------------------
def validate_live_prediction_inputs(
    release_year: int,
    rating: str,
    country: str,
    director: str,
    cast: str,
    description: str,
    valid_ratings: list[str],
    min_year: int = 1888,
    max_year: int | None = None,
    min_description_length: int = 20,
) -> list[str]:
    """Validate every Live Prediction form field. Returns a list of
    human-readable error messages (empty list = all inputs are valid).

    1888 is the earliest known year of a surviving motion picture (Roundhay
    Garden Scene), used as a sane absolute floor for Release Year.
    """
    if max_year is None:
        max_year = datetime.now().year

    errors: list[str] = []

    if release_year is None or not (min_year <= release_year <= max_year):
        errors.append(f"Release Year must be between {min_year} and {max_year}.")

    if not rating or rating not in valid_ratings:
        errors.append("Please select a valid Rating from the list.")

    if not country or not country.strip():
        errors.append("Country cannot be empty.")

    if not director or not director.strip():
        errors.append("Director cannot be empty.")

    if not cast or not cast.strip():
        errors.append("Cast cannot be empty.")

    if not description or len(description.strip()) < min_description_length:
        errors.append(f"Description must contain at least {min_description_length} characters.")

    return errors


# ---------------------------------------------------------------------------
# Live Prediction — confidence indicator
# ---------------------------------------------------------------------------
def confidence_level(confidence: float | None) -> tuple[str, str]:
    """Map a probability in [0, 1] to a simple text confidence indicator."""
    if confidence is None:
        return "Unknown", "Unknown Confidence"
    if confidence >= 0.90:
        return "High", "High Confidence"
    if confidence >= 0.70:
        return "Medium", "Medium Confidence"
    return "Low", "Low Confidence"


# ---------------------------------------------------------------------------
# Live Prediction — simple, honest, rule-based explanation (NOT SHAP)
# ---------------------------------------------------------------------------
_READABLE_FEATURE_NAMES: dict[str, str] = {
    "release_year": "release year",
    "content_age": "content age",
    "number_of_countries": "number of countries",
    "number_of_genres": "number of genres",
    "number_of_cast_members": "cast size",
    "number_of_directors": "number of directors",
}


def build_rule_based_explanation(
    input_row: dict,
    explanation_stats: dict,
    predicted_label: str,
    numeric_features: list[str],
) -> list[str]:
    """Build a short, transparent explanation by comparing this title's own
    engineered feature values to each class's historical average (computed
    once in train_models.py and passed in via `explanation_stats`).

    This is deliberately NOT SHAP, LIME, or any per-instance gradient
    attribution — it's a plain nearest-mean comparison anyone could
    reproduce by hand from the two numbers involved, so it never overstates
    its own precision.
    """
    movie_means = explanation_stats.get("Movie", {})
    tv_means = explanation_stats.get("TV Show", {})
    other_label = "Movie" if predicted_label == "TV Show" else "TV Show"

    leans_predicted: list[str] = []
    leans_other: list[str] = []

    for feature in numeric_features:
        if feature not in movie_means or feature not in tv_means:
            continue
        value = input_row.get(feature)
        if value is None:
            continue
        distance_to_movie = abs(value - movie_means[feature])
        distance_to_tv = abs(value - tv_means[feature])
        closer_to = "Movie" if distance_to_movie < distance_to_tv else "TV Show"
        label = _READABLE_FEATURE_NAMES.get(feature, feature)
        (leans_predicted if closer_to == predicted_label else leans_other).append(label)

    lines: list[str] = []
    if leans_predicted:
        verb = "is" if len(leans_predicted) == 1 else "are"
        lines.append(
            f"This title's **{', '.join(leans_predicted)}** {verb} closer to the typical "
            f"**{predicted_label}** average in this dataset."
        )
    if leans_other:
        verb = "is" if len(leans_other) == 1 else "are"
        lines.append(
            f"Its **{', '.join(leans_other)}** {verb} actually closer to the typical "
            f"**{other_label}** average — a signal the model weighed against the final prediction."
        )
    if not lines:
        lines.append(
            "This title's numeric features sit roughly midway between both classes' historical "
            "averages — the description text likely played the larger role in this prediction."
        )
    return lines


# ---------------------------------------------------------------------------
# Live Prediction — export as CSV / PDF
# ---------------------------------------------------------------------------
def prediction_record_to_csv_bytes(record: dict) -> bytes:
    """Serialize a single prediction record dict to CSV bytes for download."""
    return pd.DataFrame([record]).to_csv(index=False).encode("utf-8")


def _wrap_text(text: str, width: int = 95) -> list[str]:
    return textwrap.wrap(text, width=width) or [""]


def prediction_record_to_pdf_bytes(
    record: dict,
    report_title: str = "Netflix Content Intelligence Dashboard — Prediction Report",
) -> bytes:
    """Render a single prediction record as a simple one-page PDF report
    using reportlab, styled with the Netflix red/black palette. Returns raw
    PDF bytes suitable for `st.download_button`.
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.pdfgen import canvas

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 1 * inch

    # Title bar
    c.setFillColorRGB(0.898, 0.035, 0.078)  # Netflix red #E50914
    c.setFont("Helvetica-Bold", 16)
    c.drawString(1 * inch, y, report_title)
    y -= 0.35 * inch

    c.setFillColorRGB(0.3, 0.3, 0.3)
    c.setFont("Helvetica", 9)
    c.drawString(1 * inch, y, "Generated by the Netflix Content Intelligence Dashboard — Live Prediction page")
    y -= 0.35 * inch

    c.setStrokeColorRGB(0.898, 0.035, 0.078)
    c.line(1 * inch, y, width - 1 * inch, y)
    y -= 0.35 * inch

    c.setFont("Helvetica-Bold", 11)
    c.setFillColorRGB(0, 0, 0)
    for key, value in record.items():
        c.setFont("Helvetica-Bold", 10)
        c.drawString(1 * inch, y, f"{key}:")
        y -= 0.22 * inch
        c.setFont("Helvetica", 10)
        for chunk in _wrap_text(str(value)):
            c.drawString(1.15 * inch, y, chunk)
            y -= 0.2 * inch
            if y < 1 * inch:
                c.showPage()
                y = height - 1 * inch
                c.setFont("Helvetica", 10)
        y -= 0.08 * inch

    c.save()
    buffer.seek(0)
    return buffer.getvalue()
