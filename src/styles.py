"""Custom CSS for the Legal Agent UI.

``get_css()`` returns a ``<style>`` block that is injected once via
``st.markdown(..., unsafe_allow_html=True)``. The design language: dark navy
sidebar, clean white content-first main area, amber accents, system font stack,
no Streamlit chrome, no emojis.
"""

from __future__ import annotations

NAVY = "#1B2A4A"
AMBER = "#D4A843"

_CSS = """
<style>
/* ---- Font stack (system fonts, no external dependency) ---- */
html, body, [class*="css"], [data-testid="stAppViewContainer"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

/* ---- Hide only the "Made with Streamlit" footer ---- */
/* Streamlit's header, toolbar, and sidebar controls are left untouched so the
   sidebar can always be collapsed and reopened. */
footer { visibility: hidden; }

/* ---- Main area: clean white, centered, not edge-to-edge ---- */
[data-testid="stAppViewContainer"] { background: #FFFFFF; }
.block-container,
[data-testid="stMainBlockContainer"] {
    max-width: 840px;
    margin: 0 auto;
    padding-top: 1.75rem;
    padding-bottom: 7rem;
}

/* ---- Custom brand header (sidebar) ---- */
.brand {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    padding: 0.25rem 0 0.9rem 0;
}
.brand-logo svg { width: 38px; height: 38px; display: block; }
.brand-name {
    color: #FFFFFF;
    font-size: 1.3rem;
    font-weight: 700;
    letter-spacing: -0.01em;
    line-height: 1.1;
}
.brand-sub {
    color: #9FB0C9;
    font-size: 0.75rem;
    font-weight: 400;
    margin-top: 2px;
}

/* ---- Main header ---- */
.main-title {
    color: #1B2A4A;
    font-size: 1.7rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin: 0 0 0.25rem 0;
}
.info-bar {
    color: #5A6580;
    font-size: 0.9rem;
    background: #F7F8FA;
    border: 1px solid #ECEEF2;
    border-left: 3px solid #D4A843;
    border-radius: 8px;
    padding: 0.6rem 0.9rem;
    margin-bottom: 1.25rem;
}

/* ---- Sidebar: dark navy ---- */
[data-testid="stSidebar"] {
    background: #1B2A4A;
    border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] * { color: #D7DEEA; }
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .section-label {
    color: #AFBBD0 !important;
}
[data-testid="stSidebar"] .section-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin: 0.5rem 0 0.35rem 0;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.10);
    margin: 0.9rem 0;
}

/* ---- Sidebar buttons ---- */
[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    text-align: left;
    background: rgba(255,255,255,0.05);
    color: #E7ECF4;
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 8px;
    padding: 0.55rem 0.8rem;
    font-size: 0.86rem;
    font-weight: 500;
    line-height: 1.35;
    transition: all 0.15s ease;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(212,168,67,0.14);
    border-color: #D4A843;
    color: #FFFFFF;
}
/* New Chat = primary (amber) */
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: #D4A843;
    color: #1B2A4A;
    border: 1px solid #D4A843;
    font-weight: 600;
    text-align: center;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
    background: #E4BA55;
    color: #1B2A4A;
}

/* ---- Sidebar selectbox ---- */
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: rgba(255,255,255,0.06);
    border-color: rgba(255,255,255,0.14);
    border-radius: 8px;
    color: #E7ECF4;
}

/* ---- Sidebar disclaimer ---- */
.sidebar-disclaimer {
    color: #8494AD;
    font-size: 0.72rem;
    line-height: 1.45;
    border-top: 1px solid rgba(255,255,255,0.10);
    padding-top: 0.8rem;
    margin-top: 0.4rem;
}

/* ---- Chat messages ---- */
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] { display: none !important; }

[data-testid="stChatMessage"] {
    display: block;
    background: #FFFFFF;
    border: 1px solid #E7E9EE;
    border-left: 3px solid #D4A843;
    border-radius: 10px;
    padding: 0.85rem 1.15rem;
    margin: 0.55rem 0;
    max-width: 88%;
    box-shadow: 0 1px 2px rgba(27,42,74,0.05);
}
[data-testid="stChatMessage"] p { color: #24304A; line-height: 1.6; }
/* User messages: light background, right-aligned, no amber accent */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: #F4F6F9;
    border: 1px solid #E3E7EE;
    margin-left: auto;
}

/* ---- Chat input ---- */
[data-testid="stBottomBlockContainer"] {
    max-width: 840px;
    margin: 0 auto;
    background: #FFFFFF;
}
[data-testid="stChatInput"] {
    border: 1px solid #D3D8E1;
    border-radius: 12px;
    background: #FFFFFF;
    box-shadow: 0 1px 3px rgba(27,42,74,0.06);
}
[data-testid="stChatInput"]:focus-within {
    border-color: #D4A843;
    box-shadow: 0 0 0 2px rgba(212,168,67,0.18);
}
[data-testid="stChatInput"] textarea { color: #24304A; }

/* ---- Sources expander ---- */
[data-testid="stExpander"] {
    border: 1px solid #E7E9EE;
    border-radius: 10px;
    background: #FBFBFC;
    box-shadow: none;
}
[data-testid="stExpander"] summary {
    font-size: 0.85rem;
    font-weight: 600;
    color: #3A4763;
    padding: 0.6rem 0.9rem;
}
[data-testid="stExpander"] summary:hover { color: #1B2A4A; }
.source-item { padding: 0.15rem 0 0.35rem 0; }
.source-title { font-size: 0.9rem; font-weight: 600; color: #1B2A4A; }
.source-meta { font-size: 0.74rem; color: #8A93A6; margin: 2px 0 4px 0; }
.source-snippet { font-size: 0.85rem; color: #45506B; line-height: 1.5; }
.source-divider { border-top: 1px solid #ECEEF2; margin: 0.7rem 0; }

/* ---- Clean error / empty state ---- */
.app-error {
    color: #8A3A3A;
    background: #FBF3F3;
    border: 1px solid #E6D3D3;
    border-radius: 8px;
    padding: 0.6rem 0.9rem;
    font-size: 0.88rem;
}
</style>
"""


def get_css() -> str:
    """Return the app's custom CSS as an injectable ``<style>`` block."""
    return _CSS
