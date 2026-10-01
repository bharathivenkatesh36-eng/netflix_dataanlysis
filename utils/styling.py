"""
utils/styling.py
=================
Shared Netflix-inspired visual theme: CSS injection, KPI cards, page headers,
and a consistent Plotly color palette/template used across every page.
"""

from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------------------
# Brand palette
# ---------------------------------------------------------------------------
NETFLIX_RED = "#E50914"
NETFLIX_DARK_RED = "#B20710"
NETFLIX_PINK = "#F5576C"
NETFLIX_BLACK = "#070707"
NETFLIX_DARKER = "#030303"
NETFLIX_CARD = "#090909"
NETFLIX_GRAY = "#651016"
NETFLIX_TEXT_MUTED = "#B3B3B3"
NETFLIX_WHITE = "#FFFFFF"

# Sequential/qualitative palette for charts (reds/pinks fading to muted grays,
# echoing the reference dashboard screenshots)
PLOTLY_COLORWAY = [
    "#E50914", "#F5576C", "#B20710", "#FF6B6B", "#831010",
    "#FFC1C1", "#7A7A7A", "#D9534F", "#4A0404", "#FF9A9A",
]
PLOTLY_CONTINUOUS_SCALE = "Reds"

NAV_GROUPS = [
    ("Workspace", [("Home", "pages/1_Home.py", "home"), ("Dataset Analysis", "pages/2_Dataset_Analysis.py", "database")]),
    ("Prepare & Explore", [("Data Preprocessing", "pages/3_Data_Preprocessing.py", "tune"), ("Exploratory Data Analysis", "pages/4_EDA.py", "bar_chart"), ("Key Insights", "pages/10_Insights.py", "lightbulb")]),
    ("Machine Learning", [("Supervised ML", "pages/5_Supervised_ML.py", "psychology"), ("Model Performance", "pages/8_Model_Performance.py", "military_tech"), ("Unsupervised ML", "pages/6_Unsupervised_ML.py", "hub"), ("Recommendation System", "pages/7_Recommendation_System.py", "auto_awesome"), ("Live Prediction", "pages/11_Live_Prediction.py", "target")]),
    ("Project", [("About", "pages/9_About.py", "info")]),
]


def render_sidebar() -> None:
    """Render a compact, grouped product-style sidebar navigation."""
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="sidebar-wordmark">NETFLIX</div>
                <div class="sidebar-product">Content Intelligence</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for group_name, links in NAV_GROUPS:
            st.markdown(f'<div class="sidebar-group">{group_name}</div>', unsafe_allow_html=True)
            for label, path, icon in links:
                st.page_link(path, label=label, icon=f":material/{icon}:")


def apply_netflix_theme() -> None:
    """Inject the shared black-and-red dashboard theme. Call once per page."""
    st.markdown(
        f"""
        <style>
        :root {{
            --netflix-red: {NETFLIX_RED};
            --netflix-red-dark: {NETFLIX_DARK_RED};
            --netflix-black: {NETFLIX_BLACK};
            --netflix-panel: {NETFLIX_CARD};
            --netflix-line: #343434;
        }}

        .stApp {{
            background-color: {NETFLIX_BLACK};
            background-image:
                linear-gradient(155deg, rgba(229, 9, 20, 0.16) 0, transparent 18%, transparent 72%, rgba(229, 9, 20, 0.10) 100%),
                repeating-linear-gradient(132deg, transparent 0, transparent 96px, rgba(229, 9, 20, 0.11) 97px, transparent 99px),
                repeating-linear-gradient(43deg, transparent 0, transparent 122px, rgba(255, 255, 255, 0.04) 123px, transparent 125px);
            color: {NETFLIX_WHITE};
        }}

        .main .block-container {{
            max-width: 1500px;
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }}

        section[data-testid="stSidebar"] {{
            background-color: {NETFLIX_DARKER};
            border-right: 2px solid {NETFLIX_DARK_RED};
            min-width: 300px;
            max-width: 300px;
            width: 300px;
        }}
        section[data-testid="stSidebar"] > div:first-child {{ padding: 1.2rem 0.9rem; }}
        div[data-testid="stSidebarNav"] {{ display: none; }}
        section[data-testid="stSidebar"] * {{
            color: {NETFLIX_WHITE} !important;
        }}
        .sidebar-brand {{ padding: 0.35rem 0.35rem 1.4rem; border-bottom: 1px solid #292929; margin-bottom: 1.2rem; }}
        .sidebar-wordmark {{ color: {NETFLIX_RED}; font-family: Impact, 'Arial Narrow', sans-serif; font-size: 2rem; line-height: 1; letter-spacing: 0.03em; }}
        .sidebar-product {{ color: {NETFLIX_TEXT_MUTED}; font-size: 0.78rem; margin-top: 0.35rem; letter-spacing: 0.02em; }}
        .sidebar-group {{ color: #707070; font-size: 0.68rem; font-weight: 700; letter-spacing: 0.11em; margin: 1rem 0 0.35rem 0.35rem; text-transform: uppercase; }}
        section[data-testid="stSidebar"] a {{ border-radius: 5px; padding: 0.45rem 0.55rem; margin: 0.1rem 0; transition: background-color 0.2s ease; }}
        section[data-testid="stSidebar"] a:hover {{ background: #1d1d1d; }}
        section[data-testid="stSidebar"] a[aria-current="page"] {{ background: rgba(229, 9, 20, 0.16); border-left: 3px solid {NETFLIX_RED}; }}

        h1, h2, h3, h4 {{
            color: {NETFLIX_RED} !important;
            font-family: 'Arial Narrow', 'Trebuchet MS', sans-serif;
            font-weight: 700 !important;
            letter-spacing: 0;
        }}
        p, li, span, label, div {{
            font-family: 'Trebuchet MS', Arial, sans-serif;
        }}
        .material-symbols-outlined {{ font-family: 'Material Symbols Outlined'; font-weight: normal; font-style: normal; font-size: 1.05rem; line-height: 1; vertical-align: -0.15em; }}

        [data-testid="stHeader"] {{ background: transparent; }}
        [data-testid="stToolbar"] {{ visibility: hidden; }}

        /* The reference dashboard uses black framed panels with red edges. */
        div[data-testid="stVerticalBlockBorderWrapper"],
        div[data-testid="stExpander"],
        div[data-testid="stMetric"] {{
            background: rgba(3, 3, 3, 0.94);
            border-color: var(--netflix-line);
            box-shadow: 0 8px 22px rgba(0, 0, 0, 0.34);
        }}

        /* Buttons */
        .stButton > button, .stDownloadButton > button {{
            background: {NETFLIX_RED};
            color: {NETFLIX_WHITE};
            border: 1px solid #ff3942;
            border-radius: 6px;
            font-weight: 600;
            padding: 0.5rem 1.5rem;
            transition: background-color 0.2s ease, transform 0.2s ease;
        }}
        .stButton > button:hover, .stDownloadButton > button:hover {{
            background-color: {NETFLIX_DARK_RED};
            color: {NETFLIX_WHITE};
            border-color: {NETFLIX_RED};
            transform: translateY(-1px);
        }}

        /* Tabs */
        .stTabs [data-baseweb="tab"] {{
            color: {NETFLIX_TEXT_MUTED};
            font-weight: 600;
        }}
        .stTabs [aria-selected="true"] {{
            color: {NETFLIX_RED} !important;
            border-bottom-color: {NETFLIX_RED} !important;
        }}

        /* Metrics */
        div[data-testid="stMetric"] {{
            border: 1px solid var(--netflix-line);
            border-top: 3px solid {NETFLIX_RED};
            border-radius: 8px;
            padding: 0.9rem 1rem;
            min-height: 112px;
            box-sizing: border-box;
        }}
        div[data-testid="stMetricLabel"] {{
            color: {NETFLIX_RED} !important;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }}
        div[data-testid="stMetricValue"] {{
            color: {NETFLIX_WHITE} !important;
            font-weight: 800;
        }}

        /* Dataframes / tables */
        .stDataFrame {{
            border: 1px solid var(--netflix-line);
            border-radius: 8px;
            overflow: hidden;
        }}

        /* Inputs */
        .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"], .stMultiSelect div[data-baseweb="select"] {{
            background-color: {NETFLIX_CARD} !important;
            color: {NETFLIX_WHITE} !important;
            border-color: var(--netflix-line) !important;
            border-radius: 5px;
        }}

        /* Expander */
        .streamlit-expanderHeader {{
            background-color: {NETFLIX_CARD};
            color: {NETFLIX_WHITE} !important;
            border-radius: 6px;
        }}

        /* Divider */
        hr {{
            border-color: #2a2a2a;
        }}

        /* Custom classes */
        .netflix-title {{
            color: {NETFLIX_RED};
            font-size: 2.45rem;
            font-weight: 800;
            letter-spacing: 0;
            margin-bottom: 0;
        }}
        .netflix-subtitle {{
            color: {NETFLIX_TEXT_MUTED};
            font-size: 1.05rem;
            margin-top: 0.2rem;
        }}
        .section-rule {{ height: 1px; background: #2a2a2a; margin: 0.8rem 0 1.8rem; }}
        .home-hero {{ background: linear-gradient(110deg, rgba(20, 20, 20, 0.95), rgba(8, 8, 8, 0.9)); border-left: 3px solid {NETFLIX_RED}; border-bottom: 1px solid #2b2b2b; padding: 1.8rem 0 1.7rem 1.3rem; margin: 0 0 1.4rem; }}
        .home-hero .hero-copy {{ color: {NETFLIX_WHITE}; font-size: 1.02rem; font-weight: 500; line-height: 1.6; max-width: 780px; margin-left: 0.4rem; }}
        .section-head {{ color: {NETFLIX_RED} !important; margin: 1.2rem 0 1rem; font-size: clamp(2.1rem, 2.4vw, 3.1rem); line-height: 1.15; letter-spacing: -0.02em; }}
        .quick-card {{ background: rgba(9, 9, 9, 0.94); border: 1px solid #343434; border-top: 3px solid {NETFLIX_DARK_RED}; border-radius: 10px; min-height: 210px; padding: 1.2rem 1.15rem; margin-bottom: 0.7rem; box-shadow: 0 8px 18px rgba(0, 0, 0, 0.18); display: flex; flex-direction: column; justify-content: flex-start; }}
        .quick-card .quick-title {{ color: {NETFLIX_RED}; font-size: clamp(1.5rem, 1.6vw, 2rem); font-weight: 700; letter-spacing: -0.03em; margin: 0 0 0.6rem; line-height: 1.2; text-transform: lowercase; }}
        .quick-card .quick-description {{ color: {NETFLIX_TEXT_MUTED}; font-size: 0.84rem; line-height: 1.5; margin-bottom: 0.8rem; min-height: 92px; }}
        .quick-card .quick-link {{ display: inline-flex; align-items: center; gap: 0.45rem; color: #d9d9d9; font-size: 0.78rem; font-weight: 600; margin-top: auto; opacity: 0.9; }}
        .quick-card .material-symbols-outlined {{ font-size: 1rem; color: #f0f0f0; }}
        .insight-stat {{ background: rgba(9, 9, 9, 0.94); border: 1px solid #343434; border-left: 3px solid {NETFLIX_RED}; border-radius: 8px; min-height: 132px; padding: 1rem 1.1rem; }}
        .insight-stat-label {{ color: {NETFLIX_TEXT_MUTED}; font-size: 0.76rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; }}
        .insight-stat-value {{ color: {NETFLIX_WHITE}; font-size: 1.45rem; font-weight: 800; margin: 0.55rem 0 0.35rem; overflow-wrap: anywhere; }}
        .insight-stat-detail {{ color: #888888; font-size: 0.78rem; line-height: 1.35; }}
        .technical-card {{ min-height: 165px; }}
        .technical-card p {{ color: {NETFLIX_TEXT_MUTED}; font-size: 0.88rem; line-height: 1.55; }}
        .about-hero {{ background: linear-gradient(110deg, rgba(20, 20, 20, 0.96), rgba(8, 8, 8, 0.9)); border-left: 4px solid {NETFLIX_RED}; border-bottom: 1px solid #2b2b2b; padding: 2rem 2.2rem; margin-bottom: 1.8rem; }}
        .about-hero .hero-kicker {{ color: {NETFLIX_RED}; font-size: 0.75rem; font-weight: 800; letter-spacing: 0.15em; }}
        .about-hero h1 {{ color: {NETFLIX_WHITE} !important; font-size: clamp(2rem, 4vw, 3.2rem); margin: 0.5rem 0; }}
        .about-hero p {{ color: {NETFLIX_TEXT_MUTED}; font-size: 1rem; line-height: 1.55; max-width: 760px; margin: 0; }}
        .about-principle {{ display: flex; flex-direction: column; gap: 0.25rem; background: rgba(9, 9, 9, 0.8); border-left: 3px solid {NETFLIX_DARK_RED}; padding: 0.8rem 1rem; margin: 0.65rem 0; }}
        .about-principle strong {{ color: {NETFLIX_WHITE}; }}
        .about-principle span {{ color: {NETFLIX_TEXT_MUTED}; font-size: 0.88rem; line-height: 1.45; }}
        .about-stack {{ padding: 0.35rem 1.1rem; }}
        .about-stack div {{ display: flex; justify-content: space-between; gap: 1rem; padding: 0.75rem 0; border-bottom: 1px solid #292929; }}
        .about-stack div:last-child {{ border-bottom: 0; }}
        .about-stack b {{ color: {NETFLIX_WHITE}; font-size: 0.86rem; }}
        .about-stack span {{ color: {NETFLIX_TEXT_MUTED}; font-size: 0.82rem; text-align: right; }}
        .workflow-card {{ background: rgba(9, 9, 9, 0.94); border: 1px solid #343434; border-top: 3px solid {NETFLIX_DARK_RED}; border-radius: 8px; min-height: 175px; padding: 1rem; margin-bottom: 1rem; }}
        .workflow-number {{ color: {NETFLIX_RED}; font-size: 0.78rem; font-weight: 800; letter-spacing: 0.1em; }}
        .workflow-card h4 {{ color: {NETFLIX_WHITE} !important; font-size: 0.95rem; margin: 0.65rem 0 0.45rem; }}
        .workflow-card p, .methodology-card p {{ color: {NETFLIX_TEXT_MUTED}; font-size: 0.82rem; line-height: 1.5; }}
        .methodology-card {{ min-height: 170px; }}
        .netflix-badge {{
            display: inline-block;
            background-color: {NETFLIX_RED};
            color: white;
            padding: 0.15rem 0.6rem;
            border-radius: 3px;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.5px;
            margin-right: 0.4rem;
        }}
        .netflix-card {{
            background: rgba(3, 3, 3, 0.94);
            border: 1px solid var(--netflix-line);
            border-top: 3px solid {NETFLIX_DARK_RED};
            border-radius: 9px;
            padding: 1.2rem 1.4rem;
            margin-bottom: 1rem;
            box-shadow: 0 5px 16px rgba(0, 0, 0, 0.24);
        }}
        .netflix-card h4 {{
            color: {NETFLIX_RED} !important;
            margin-top: 0;
        }}
        .insight-box {{
            background-color: {NETFLIX_CARD};
            border-left: 4px solid {NETFLIX_RED};
            border-radius: 6px;
            padding: 0.9rem 1.1rem;
            margin: 0.8rem 0 1.4rem 0;
            color: {NETFLIX_TEXT_MUTED};
            font-size: 0.95rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    render_sidebar()


def page_header(title: str, subtitle: str = "", icon_name: str = "dashboard") -> None:
    """Render a consistent page header using Streamlit Material Symbols."""
    st.markdown(
        f"## :material/{icon_name}: {title}",
    )
    if subtitle:
        st.markdown(f'<div class="netflix-subtitle">{subtitle}</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)


def insight_box(text: str) -> None:
    """Render a small highlighted explanation/insight box under a chart."""
    st.markdown(f'<div class="insight-box"><span class="material-symbols-outlined">lightbulb</span>{text}</div>', unsafe_allow_html=True)


def render_kpi_cards(cards: list[dict]) -> None:
    """Render a row of KPI metric cards.

    `cards` is a list of dicts: {"label": str, "value": str, "delta": str (optional)}
    """
    cols = st.columns(len(cards))
    for col, card in zip(cols, cards):
        with col:
            st.metric(label=card["label"], value=card["value"], delta=card.get("delta"))


def netflix_plotly_layout(fig, title: str | None = None, height: int = 440):
    """Apply the shared dark Netflix theme to any Plotly figure."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=NETFLIX_BLACK,
        plot_bgcolor=NETFLIX_BLACK,
        font=dict(family="Trebuchet MS, Arial, sans-serif", color=NETFLIX_WHITE, size=12),
        colorway=PLOTLY_COLORWAY,
        title=title if title else fig.layout.title.text,
        title_font=dict(size=15, color=NETFLIX_WHITE),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=NETFLIX_TEXT_MUTED)),
        hoverlabel=dict(bgcolor="#181818", bordercolor=NETFLIX_RED, font=dict(color=NETFLIX_WHITE)),
        height=height,
        margin=dict(l=55, r=25, t=52, b=45),
    )
    fig.update_xaxes(gridcolor="#292929", zerolinecolor="#292929", linecolor="#343434", tickfont=dict(color=NETFLIX_TEXT_MUTED))
    fig.update_yaxes(gridcolor="#292929", zerolinecolor="#292929", linecolor="#343434", tickfont=dict(color=NETFLIX_TEXT_MUTED))
    return fig


def render_insight_cards(insights: list[str]) -> None:
    """Render a list of one-line findings as Netflix-styled cards (Insights page)."""
    for text in insights:
        st.markdown(
            f"""
            <div class="netflix-card" style="display:flex; align-items:center; padding:0.9rem 1.2rem;">
                <span style="color:#E50914; font-size:1.3rem; margin-right:0.8rem;">▸</span>
                <span style="color:#FFFFFF; font-size:0.98rem;">{text}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
