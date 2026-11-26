"""Komponen tampilan bersama untuk seluruh halaman Streamlit."""

import streamlit as st

from src.data_loader import FEATURE_INFO


def apply_style() -> None:
    """Apply lightweight shared themes and navigation/page transitions."""
    if st.session_state.dark_mode_preference:
        background, surface, sidebar = "#101722", "rgba(28, 40, 56, .78)", "#151f2d"
        text, muted, border, accent = "#e6edf5", "#a3b1c2", "rgba(160, 180, 205, .20)", "#9bc5f0"
        menu_bg, menu_text, menu_hover = "#1c2838", "#e6edf5", "#2b4058"
        field_bg = "#1c2838"
        tooltip_bg, tooltip_text = "#26364a", "#ffffff"
        theme_icon_filter = "none"
        icon_filter = "brightness(0) saturate(100%) invert(78%) sepia(16%) saturate(810%) hue-rotate(177deg) brightness(102%)"
        header, scheme = "rgba(16, 23, 34, .90)", "dark"
        dataframe_theme = """
        [data-testid="stDataFrame"] {
            --gdg-bg-cell: #182333 !important;
            --gdg-bg-header: #202e40 !important;
            --gdg-bg-header-has-focus: #2b4058 !important;
            --gdg-bg-bubble: #2b4058 !important;
            --gdg-text-dark: #e6edf5 !important;
            --gdg-text-medium: #bdc9d7 !important;
            --gdg-text-light: #a3b1c2 !important;
            --gdg-text-group-header: #e6edf5 !important;
            --gdg-border-color: #35465b !important;
            --gdg-link-color: #9bc5f0 !important;
        }
        [data-testid="stDataFrame"] canvas { filter: invert(.88) hue-rotate(180deg); }
        """
    else:
        background, surface, sidebar = "#f4f7fb", "rgba(255, 255, 255, .78)", "#edf3f9"
        text, muted, border, accent = "#26364a", "#64748b", "rgba(105, 133, 165, .20)", "#244e78"
        menu_bg, menu_text, menu_hover = "#ffffff", "#26364a", "#edf3f9"
        field_bg = "#ffffff"
        tooltip_bg, tooltip_text = "#26364a", "#ffffff"
        # Keep the textured moon emoji, but remove its bright yellow coloring.
        theme_icon_filter = "grayscale(1) brightness(.62)"
        icon_filter = "brightness(0) saturate(100%) invert(35%) sepia(13%) saturate(1395%) hue-rotate(169deg) brightness(92%)"
        header, scheme = "rgba(244, 247, 251, .90)", "light"
        dataframe_theme = "[data-testid=\"stDataFrame\"] canvas { filter: none; }"

    st.markdown(
        f"""
        <style>
        :root {{ color-scheme: {scheme}; }}
        .stApp {{ background: radial-gradient(ellipse at 12% 0%, {accent}18, transparent 38%), {background}; color: {text}; }}
        [data-testid="stAppViewContainer"] {{ background: transparent; }}
        [data-testid="stHeader"] {{ background: {header}; backdrop-filter: blur(14px); }}
        [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
        [data-testid="collapsedControl"] {{ display: none !important; }}
        .st-key-glass_navigation {{
            position: sticky; top: .35rem; z-index: 999; padding: .55rem .7rem;
            margin: 0 0 1.25rem; border: 1px solid {border}; border-radius: 18px;
            background: {surface}; backdrop-filter: blur(18px) saturate(145%);
            -webkit-backdrop-filter: blur(18px) saturate(145%);
            box-shadow: 0 8px 28px rgba(30, 55, 85, .09);
        }}
        .st-key-glass_navigation [data-testid="stPageLink"] a {{
            color: {text} !important; border-radius: 11px; padding: .42rem .55rem;
            transition: background-color .16s ease, transform .16s ease;
        }}
        .st-key-glass_navigation [data-testid="stPageLink"] a:hover {{
            background: {accent}18; transform: translateY(-1px);
        }}
        .st-key-glass_navigation [data-testid="stPageLink"] a[aria-current="page"] {{
            color: {accent} !important; background: {accent}18; font-weight: 650;
        }}
        .st-key-glass_navigation [data-testid="stButton"] button {{
            width: 2.65rem; height: 2.65rem; padding: 0; margin-left: auto;
            border: 1px solid {border}; border-radius: 50%;
            background: {accent}18; color: {text} !important;
            transition: background-color .16s ease, transform .16s ease;
        }}
        .st-key-glass_navigation [data-testid="stButton"] button:hover {{
            background: {accent}30; transform: rotate(-8deg) scale(1.04);
            border-color: {accent};
        }}
        .st-key-glass_navigation [data-testid="stButton"] button span {{
            filter: {theme_icon_filter};
        }}
        [data-testid="stMain"] .stMainBlockContainer {{ animation: page-enter .22s ease-out both; }}
        @keyframes page-enter {{ from {{ opacity: .72; transform: translateY(6px); }} to {{ opacity: 1; transform: translateY(0); }} }}
        h1, h2, h3, h4 {{ color: {text}; letter-spacing: -0.025em; }}
        p, label, li, [data-testid="stCaptionContainer"] {{ color: {text}; }}
        input, textarea {{ color: {text} !important; }}
        [data-testid="stNumberInput"] input, [data-testid="stTextInput"] input,
        [data-testid="stTextArea"] textarea {{
            color: {text} !important; background-color: {field_bg} !important;
            -webkit-text-fill-color: {text} !important; caret-color: {text} !important;
        }}
        [data-testid="stNumberInput"] button {{ color: {text} !important; background-color: {field_bg} !important; }}
        div[data-baseweb="select"] > div * {{ color: {text} !important; }}
        div[data-baseweb="select"] > div {{ background: {surface} !important; border-color: {border} !important; }}
        div[data-baseweb="select"] [data-baseweb="tag"],
        div[data-baseweb="select"] [data-baseweb="tag"] * {{ color: #ffffff !important; }}
        [data-baseweb="popover"], [data-baseweb="menu"], [role="listbox"] {{
            background-color: {menu_bg} !important; color: {menu_text} !important;
        }}
        [role="option"] {{ color: {menu_text} !important; background-color: {menu_bg} !important; }}
        [role="option"]:hover, [role="option"][aria-selected="true"] {{ background-color: {menu_hover} !important; }}
        [data-testid="stMarkdownContainer"] a {{ color: {accent}; }}
        [data-testid="stMetric"] {{
            border: 1px solid {border}; border-radius: 16px; padding: 16px;
            background: {surface}; backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
            box-shadow: 0 8px 24px rgba(30, 55, 85, .07);
            transition: transform .2s ease, box-shadow .2s ease;
        }}
        [data-testid="stMetric"]:hover {{ transform: translateY(-2px); box-shadow: 0 12px 30px rgba(30, 55, 85, .12); }}
        [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {{ color: {accent}; }}
        [data-testid="stTooltipIcon"] {{ color: {accent} !important; background: transparent !important; }}
        [data-testid="stTooltipIcon"] svg {{ filter: {icon_filter}; }}
        [data-baseweb="tooltip"], [role="tooltip"], [data-testid="stTooltipContent"] {{
            color: {tooltip_text} !important; background: {tooltip_bg} !important;
            border: 1px solid {border} !important; border-radius: 8px !important;
        }}
        [data-baseweb="tooltip"] *, [role="tooltip"] *, [data-testid="stTooltipContent"] * {{
            color: {tooltip_text} !important; -webkit-text-fill-color: {tooltip_text} !important;
        }}
        [data-testid="stFormSubmitButton"] button,
        [data-testid="stFormSubmitButton"] button * {{
            color: #ffffff !important; -webkit-text-fill-color: #ffffff !important;
        }}
        div[data-testid="stDataFrame"] {{ border: 1px solid {border}; border-radius: 12px; overflow: hidden; }}
        [data-baseweb="select"] > div, [data-testid="stNumberInput"] > div,
        [data-testid="stTextInput"] > div {{ background: {surface}; border-color: {border}; }}
        {dataframe_theme}
        [data-testid="stAlert"] {{ border-radius: 12px; }}
        [data-testid="stAlert"] [data-testid="stMarkdownContainer"],
        [data-testid="stAlert"] [data-testid="stMarkdownContainer"] p {{ color: {text} !important; }}
        .muted {{ color: {muted}; font-size: 0.92rem; }}
        @media (prefers-reduced-motion: reduce) {{
            *, *::before, *::after {{ transition-duration: .01ms !important; animation-duration: .01ms !important; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def feature_name(feature: str) -> str:
    return FEATURE_INFO.get(feature, (feature.replace("_", " ").title(), ""))[0]


def show_page_intro(title: str, subtitle: str) -> None:
    st.title(title)
    st.caption(subtitle)
