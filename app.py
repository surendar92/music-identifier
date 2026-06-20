import streamlit as st
import sqlite3
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import urllib.parse

st.set_page_config(
    page_title="Music Identifier",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* ── SATOSHI FONT ── */
@import url('https://api.fontshare.com/v2/css?f[]=satoshi@400,500,600,700,800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=block');

/* Apply Satoshi to everything EXCEPT Material icon spans */
*:not([data-testid="stIconMaterial"]):not(.material-symbols-rounded),
html, body, [class*="css"] {
    font-family: 'Satoshi', 'Inter', sans-serif !important;
}

/* Ensure Material Symbols font is always used for icon elements */
[data-testid="stIconMaterial"],
.material-symbols-rounded,
[data-testid="stSidebarCollapseButton"] button span,
[data-testid="collapsedControl"] span,
[data-testid="stFileUploaderDropzone"] button span {
    font-family: 'Material Symbols Rounded' !important;
}

/* ── MAIN BACKGROUND — light green from RoomSketch ── */
.stApp { background: #f0f4ec !important; }
.main .block-container { padding: 2rem 2.5rem !important; max-width: 100% !important; }

/* ── SIDEBAR — dark green ── */
[data-testid="stSidebar"] {
    background: #162316 !important;
    border-right: none !important;
    box-shadow: 3px 0 20px rgba(0,0,0,0.25) !important;
    visibility: visible !important;
    display: block !important;
}
[data-testid="stSidebarContent"] { padding: 1.5rem 1.2rem !important; }

/* Make ALL sidebar text white */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
}

/* ── SIDEBAR COLLAPSE/EXPAND BUTTON (the arrow inside sidebar) ── */
[data-testid="stSidebarCollapseButton"] {
    visibility: visible !important;
    display: flex !important;
    opacity: 1 !important;
}
[data-testid="stSidebarCollapseButton"] button {
    background: rgba(255,255,255,0.08) !important;
    border: none !important;
    border-radius: 6px !important;
}
[data-testid="stSidebarCollapseButton"] button:hover {
    background: rgba(255,255,255,0.18) !important;
}
/* Icon inside the collapse button */
[data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"] {
    font-family: 'Material Symbols Rounded' !important;
    color: rgba(255,255,255,0.7) !important;
    font-size: 20px !important;
}

/* ── COLLAPSED CONTROL (the arrow tab when sidebar is hidden) ── */
[data-testid="collapsedControl"] {
    visibility: visible !important;
    display: flex !important;
    opacity: 1 !important;
    pointer-events: auto !important;
    z-index: 999999 !important;
    background-color: #162316 !important;
    border-radius: 0 8px 8px 0 !important;
    box-shadow: 3px 0 12px rgba(0,0,0,0.4) !important;
    padding: 12px 6px !important;
    cursor: pointer !important;
    align-items: center !important;
    justify-content: center !important;
    transition: background 0.2s !important;
}
[data-testid="collapsedControl"]:hover {
    background: #2d5a27 !important;
}
[data-testid="collapsedControl"] [data-testid="stIconMaterial"],
[data-testid="collapsedControl"] svg {
    font-family: 'Material Symbols Rounded' !important;
    fill: #ffffff !important;
    color: #ffffff !important;
    font-size: 18px !important;
    width: 18px !important;
    height: 18px !important;
}

/* ── SIDEBAR LOGO ── */
.sb-logo {
    display: flex; align-items: center; gap: 12px;
    padding-bottom: 20px; border-bottom: 1px solid rgba(255,255,255,0.12);
    margin-bottom: 20px;
}
.sb-logo-icon {
    width: 38px; height: 38px; background: #2d5a27;
    border-radius: 8px; display: flex; align-items: center;
    justify-content: center; font-size: 1.2rem;
}
.sb-logo-text { font-size: 1.2rem; font-weight: 800; color: #ffffff !important; }

.sb-section {
    font-size: 0.6rem; font-weight: 700; color: rgba(255,255,255,0.35) !important;
    letter-spacing: 3px; text-transform: uppercase; margin: 18px 0 10px;
    display: block;
}

/* ── 2x2 STATS GRID ── */
.stats-grid {
    display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 18px;
}
.stat-cell {
    background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.1);
    border-radius: 10px; padding: 12px 10px; text-align: center;
}
.stat-num { font-size: 1.25rem; font-weight: 800; color: #ffffff !important; display: block; line-height: 1.1; }
.stat-lbl { font-size: 0.58rem; color: rgba(255,255,255,0.4) !important; text-transform: uppercase; letter-spacing: 1px; margin-top: 2px; display: block; }

/* Nav items */
.sb-nav-item {
    display: flex; align-items: center; gap: 10px;
    padding: 10px 12px; border-radius: 8px; margin-bottom: 4px;
    color: rgba(255,255,255,0.65) !important; font-size: 0.88rem; font-weight: 500;
    transition: background 0.15s; cursor: pointer;
}
.sb-nav-item:hover { background: rgba(255,255,255,0.08) !important; color: white !important; }
.sb-nav-item.active {
    background: #2d5a27 !important; color: #ffffff !important; font-weight: 700;
}

/* Hide streamlit button styling in sidebar nav */
[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    color: rgba(255,255,255,0.7) !important;
    border-radius: 8px !important;
    padding: 8px 14px !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    box-shadow: none !important;
    margin-bottom: 4px !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255,255,255,0.08) !important;
    color: white !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ── PAGE HEADER — dark green bg, white text ── */
.page-header {
    background: #162316;
    border-radius: 16px; padding: 28px 32px; margin-bottom: 24px;
    box-shadow: 0 4px 20px rgba(22,35,22,0.25);
}
.page-header h1 { font-size: 1.9rem; font-weight: 800; color: #ffffff !important; margin: 0 0 4px; }
.page-header p  { color: rgba(255,255,255,0.6) !important; margin: 0; font-size: 0.9rem; }

/* ── MODE CARDS — white bg, black text ── */
.mode-card {
    background: #ffffff; border-radius: 14px; padding: 24px 20px;
    border: 2px solid #e2e8dc; text-align: center;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    height: 120px; display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    transition: all 0.2s;
}
.mode-card h3 { color: #0f1f0f !important; margin: 0 0 5px; font-size: 1.05rem; font-weight: 700; }
.mode-card p  { color: #6b7f6b !important; font-size: 0.8rem; margin: 0; }
.mode-card.active {
    border-color: #2d5a27; background: #edf3ea;
    box-shadow: 0 6px 20px rgba(45,90,39,0.15);
}
.mode-card.active h3 { color: #162316 !important; }

/* ── ALL MAIN CONTENT TEXT = BLACK ── */
.main p, .main h1, .main h2, .main h3, .main h4, .main label, 
.main [data-testid="stMarkdownContainer"] > p {
    color: #0f1f0f !important;
}

/* ── RESULT CARD — dark green bg, white text ── */
.result-card {
    background: #162316; border-radius: 16px;
    padding: 28px 32px; margin: 16px 0;
    box-shadow: 0 8px 30px rgba(22,35,22,0.3);
    position: relative; overflow: hidden;
}
.result-card::after {
    content: '🎵'; position: absolute; right: 32px; top: 50%;
    transform: translateY(-50%); font-size: 5rem; opacity: 0.08;
}
.result-label { font-size: 0.6rem; color: rgba(255,255,255,0.45) !important; letter-spacing: 4px; text-transform: uppercase; margin-bottom: 6px; display: block; }
.result-song  { font-size: 2rem; font-weight: 800; color: #ffffff !important; margin-bottom: 6px; line-height: 1.1; display: block; }
.result-meta  { font-size: 0.82rem; color: rgba(255,255,255,0.55) !important; }
.result-meta b { color: #86efac !important; }
.result-time  {
    display: inline-block; margin-top: 10px;
    background: rgba(255,255,255,0.1); color: #86efac !important;
    padding: 5px 14px; border-radius: 20px; font-size: 0.82rem; font-weight: 600;
}
.yt-btn {
    display: inline-block; margin-top: 12px; margin-left: 8px;
    background: #ffffff; color: #162316 !important; font-weight: 700;
    text-decoration: none !important; padding: 8px 18px;
    border-radius: 20px; font-size: 0.82rem;
    box-shadow: 0 3px 10px rgba(0,0,0,0.15);
}

/* ── SECTION HEADER ── */
.sec-hdr {
    font-size: 1rem; font-weight: 700; color: #0f1f0f !important;
    border-left: 4px solid #2d5a27; padding-left: 12px; margin: 20px 0 12px;
    display: block;
}

/* ── CANDIDATE ROWS — white bg, black text ── */
.cand-row {
    display: flex; align-items: center; gap: 12px;
    padding: 11px 14px; border-radius: 10px;
    background: #ffffff; margin-bottom: 6px;
    border: 1px solid #e2e8dc;
    box-shadow: 0 1px 6px rgba(0,0,0,0.04);
    transition: transform 0.15s;
}
.cand-row:hover { transform: translateX(3px); border-color: #a3c49a; }
.cand-rank { font-size: 0.7rem; color: #9ca3af !important; width: 22px; font-weight: 700; }
.cand-name { color: #0f1f0f !important; font-size: 0.88rem; font-weight: 600; flex: 1; }
.cand-score{
    color: #162316 !important; font-size: 0.8rem; font-weight: 700;
    background: #d1e8c8; padding: 3px 10px; border-radius: 8px;
}

/* ── METRICS ── */
[data-testid="metric-container"] {
    background: #ffffff !important; border-radius: 12px !important;
    border: 1px solid #e2e8dc !important; padding: 16px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04) !important;
}
[data-testid="stMetricLabel"] { color: #6b7f6b !important; font-size: 0.7rem !important; font-weight: 600 !important; letter-spacing: 1px; text-transform: uppercase; }
[data-testid="stMetricValue"] { color: #0f1f0f !important; font-size: 1.5rem !important; font-weight: 800 !important; }

/* ── BUTTONS (main area) ── */
.main .stButton > button {
    background: #162316 !important; color: #ffffff !important;
    border: none !important; border-radius: 8px !important;
    font-weight: 700 !important; font-size: 0.88rem !important;
    padding: 11px 22px !important;
    box-shadow: 0 3px 12px rgba(22,35,22,0.3) !important;
}
.main .stButton > button:hover {
    background: #2d5a27 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 18px rgba(22,35,22,0.35) !important;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: #ffffff !important; 
    border-radius: 10px !important;
    padding: 6px !important; 
    gap: 20px !important; /* 👈 Increased from 3px to 20px for clear breathing room */
    border: 1px solid #e2e8dc !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.04) !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; 
    color: #6b7f6b !important;
    border-radius: 7px !important; 
    font-size: 0.83rem !important; 
    font-weight: 600 !important;
    padding-left: 16px !important;  /* 👈 Added horizontal internal padding */
    padding-right: 16px !important; /* 👈 Added horizontal internal padding */
}
.stTabs [aria-selected="true"] {
    background: #162316 !important; 
    color: #ffffff !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: transparent !important; 
    padding-top: 1rem !important;
}
/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] {
    background: #ffffff !important;
    border-radius: 12px !important;
    border: 2px dashed #b8d4b0 !important;
    padding: 16px !important;
}

[data-testid="stFileUploaderDropzone"] {
    background-color: #f8faf6 !important;
    border-radius: 8px !important;
    border: none !important;
    padding: 28px 16px !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 8px !important;
}

/* Hide the "Drag and drop" instruction text span */
[data-testid="stFileUploaderDropzoneInstructions"] > div > span {
    display: none !important;
}

/* ── FILE UPLOADER BUTTON FIX ── */
[data-testid="stFileUploaderDropzone"] button {
    background-color: #162316 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    cursor: pointer !important;

    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 0 !important;
}

/* Hover */
[data-testid="stFileUploaderDropzone"] button:hover {
    background-color: #2d5a27 !important;
}

/* Hide every icon/material/upload glyph inside the button */
[data-testid="stFileUploaderDropzone"] button [data-testid="stIconMaterial"],
[data-testid="stFileUploaderDropzone"] button .material-symbols-rounded,
[data-testid="stFileUploaderDropzone"] button svg,
[data-testid="stFileUploaderDropzone"] button span[aria-hidden="true"] {
    display: none !important;
}

/* Keep only the actual label visible */
[data-testid="stFileUploaderDropzone"] button p,
[data-testid="stFileUploaderDropzone"] button span,
[data-testid="stFileUploaderDropzone"] button div {
    color: #ffffff !important;
    font-weight: 600 !important;
    font-family: 'Satoshi', 'Inter', sans-serif !important;
}

/* File size/format hint */
[data-testid="stFileUploaderDropzoneInstructions"] small {
    color: #6b7f6b !important;
    font-size: 0.78rem !important;
    display: block !important;
    margin-top: 6px !important;
}
/* ── DATAFRAME ── */
[data-testid="stDataFrame"] {
    border-radius: 10px !important; overflow: hidden;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    background: white !important;
}

/* ── BATCH CARD ── */
.batch-card {
    background: #ffffff; border: 1px solid #e2e8dc;
    border-radius: 12px; padding: 16px 18px; margin-bottom: 6px;
    box-shadow: 0 1px 8px rgba(0,0,0,0.04);
}
.batch-fname { font-size: 0.72rem; color: #9ca3af !important; margin-bottom: 2px; }
.batch-sname { font-size: 1.05rem; font-weight: 700; color: #0f1f0f !important; }
.batch-pos   {
    display: inline-block; margin-top: 6px;
    background: #d1f0d1; color: #145214 !important;
    padding: 3px 12px; border-radius: 8px; font-size: 0.75rem; font-weight: 600;
}

/* ── EXPANDER ── */
.streamlit-expanderHeader,
[data-testid="stExpander"] summary {
    background: #ffffff !important;
    border-radius: 10px !important;
    border: 1px solid #e2e8dc !important;
    font-weight: 600 !important;
    color: #0f1f0f !important;

    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
    padding: 12px 16px !important;
    line-height: 1.35 !important;
}

/* Remove the broken default triangle/marker text that overlaps in batch mode */
[data-testid="stExpander"] summary::-webkit-details-marker {
    display: none !important;
}
[data-testid="stExpander"] summary::marker {
    content: "" !important;
}

/* Hide any material/icon glyph Streamlit injects into expander header */
[data-testid="stExpander"] summary [data-testid="stIconMaterial"],
[data-testid="stExpander"] summary .material-symbols-rounded,
[data-testid="stExpander"] summary svg,
[data-testid="stExpander"] summary span[aria-hidden="true"] {
    display: none !important;
}

/* Make sure the expander title text sits cleanly and doesn't collide */
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span,
[data-testid="stExpander"] summary div {
    color: #0f1f0f !important;
    font-family: 'Satoshi', 'Inter', sans-serif !important;
    margin: 0 !important;
    line-height: 1.35 !important;
}
.streamlit-expanderContent,
[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    background: #f8faf6 !important;
    border: 1px solid #e2e8dc !important;
    border-top: none !important; border-radius: 0 0 10px 10px !important;
    padding: 16px !important;
}

/* Force ALL text inside an expander (markdown, bold, plain) to be dark —
   this is what makes the Spectrogram/Constellation/Histogram labels
   and any other expander text visible instead of invisible/low-contrast */
[data-testid="stExpander"] p,
[data-testid="stExpander"] span,
[data-testid="stExpander"] div,
[data-testid="stExpander"] strong,
[data-testid="stExpander"] label,
[data-testid="stExpander"] li,
[data-testid="stExpander"] h1,
[data-testid="stExpander"] h2,
[data-testid="stExpander"] h3,
[data-testid="stExpander"] h4 {
    color: #0f1f0f !important;
    font-family: 'Satoshi', 'Inter', sans-serif !important;
}

/* Dedicated sub-section label style for Spectrogram / Constellation /
   Histogram headings inside the Per-File Analysis expanders */
.batch-subhdr {
    display: block;
    font-family: 'Satoshi', 'Inter', sans-serif !important;
    font-size: 0.92rem;
    font-weight: 700;
    color: #0f1f0f !important;
    margin: 14px 0 8px;
}

/* ── PROGRESS ── */
.stProgress > div > div { background: #2d5a27 !important; border-radius: 4px !important; }

/* ── DIVIDER ── */
hr { border-color: #e2e8dc !important; }

/* ── ALERTS ── */
[data-testid="stNotification"] { border-radius: 10px !important; }
.stSuccess { background: #f0fdf0 !important; border-color: #86efac !important; }
.stWarning { background: #fffbeb !important; }
.stInfo    { background: #f0fdf4 !important; border-color: #86efac !important; }

/* ── HIDE CHROME ── */
#MainMenu { visibility: hidden; }
footer     { visibility: hidden; }

header[data-testid="stHeader"] {
    background: transparent !important;
    z-index: 999 !important;
}

/* sidebar and collapsedControl styles defined above */

/* ── DATABASE VIEW — dark constellation thumbnail cards ── */
.db-grid-label {
    font-size: 0.68rem; font-weight: 700; color: #6b7f6b !important;
    letter-spacing: 3px; text-transform: uppercase; margin: 4px 0 14px;
    display: block;
}
.db-card {
    background: #0c140c; border-radius: 12px; padding: 10px 10px 14px;
    border: 1px solid #243524;
    box-shadow: 0 2px 10px rgba(0,0,0,0.15);
    transition: transform 0.15s, box-shadow 0.15s;
}
.db-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 18px rgba(0,0,0,0.25);
}
.db-card img {
    width: 100%; display: block; border-radius: 8px; margin-bottom: 10px;
}
.db-card-name {
    font-family: 'Satoshi', 'Inter', sans-serif !important;
    font-size: 0.88rem; font-weight: 700; color: #ffffff !important;
    line-height: 1.25; margin-bottom: 3px;
    display: block; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.db-card-hashes {
    font-family: 'Satoshi', 'Inter', sans-serif !important;
    font-size: 0.74rem; font-weight: 500; color: #7d9a7d !important;
    display: block;
}
.db-search-wrap [data-testid="stTextInput"] input {
    background: #ffffff !important; border-radius: 10px !important;
}

/* ── AUDIO PLAYER ── */
audio { border-radius: 8px !important; width: 100%; }
</style>
<script>
function fixUI() {
    // 1. Hide every upload icon/material glyph inside the uploader button
    document.querySelectorAll(`
        [data-testid="stFileUploaderDropzone"] button [data-testid="stIconMaterial"],
        [data-testid="stFileUploaderDropzone"] button .material-symbols-rounded,
        [data-testid="stFileUploaderDropzone"] button svg,
        [data-testid="stFileUploaderDropzone"] button span[aria-hidden="true"]
    `).forEach(el => {
        el.style.display = 'none';
    });

    // 2. Hide the "Drag and drop" instruction text span
    document.querySelectorAll('[data-testid="stFileUploaderDropzoneInstructions"] > div > span').forEach(el => {
        el.style.display = 'none';
    });

    // 3. Fix sidebar collapse button icon color
    document.querySelectorAll('[data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"]').forEach(el => {
        el.style.color = 'rgba(255,255,255,0.8)';
        el.style.fontFamily = 'Material Symbols Rounded';
    });

    // 4. Fix collapsed control icon color
    document.querySelectorAll('[data-testid="collapsedControl"] [data-testid="stIconMaterial"]').forEach(el => {
        el.style.color = '#ffffff';
        el.style.fontFamily = 'Material Symbols Rounded';
    });

    // 5. Hide broken expander header glyphs/markers in batch analysis
    document.querySelectorAll(`
        [data-testid="stExpander"] summary [data-testid="stIconMaterial"],
        [data-testid="stExpander"] summary .material-symbols-rounded,
        [data-testid="stExpander"] summary svg,
        [data-testid="stExpander"] summary span[aria-hidden="true"]
    `).forEach(el => {
        el.style.display = 'none';
    });
}
const _observer = new MutationObserver(fixUI);
_observer.observe(document.body, { childList: true, subtree: true });
document.addEventListener('DOMContentLoaded', fixUI);
setTimeout(fixUI, 500);
setTimeout(fixUI, 1500);
</script>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════
# DATABASE
# ═══════════════════════════════════════════
@st.cache_resource
def get_db_connection():
    try:
        return sqlite3.connect("music_database.db", check_same_thread=False)
    except:
        return None

@st.cache_resource
def get_db_stats():
    conn = get_db_connection()
    if conn is None:
        return 0, 0, 0, 0
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(DISTINCT song_name) FROM hashes")
        songs = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM (SELECT DISTINCT f1,f2,delta_t FROM hashes)")
        uhash = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM hashes")
        total = cur.fetchone()[0]
        size  = os.path.getsize("music_database.db")/1024/1024 if os.path.exists("music_database.db") else 0
        return songs, uhash, total, size
    except:
        return 0, 0, 0, 0

@st.cache_data
def get_song_list_with_counts():
    """Returns [(song_name, hash_count), ...] sorted by song name."""
    conn = get_db_connection()
    if conn is None:
        return []
    try:
        cur = conn.cursor()
        cur.execute("SELECT song_name, COUNT(*) FROM hashes GROUP BY song_name ORDER BY song_name")
        return cur.fetchall()
    except:
        return []

@st.cache_data
def get_constellation_thumb(song_name, w_px=300, h_px=190):
    """
    Builds a small dark constellation-map scatter for one song using the
    real (anchor_time, f1) peak coordinates stored in the hashes table,
    and returns it as a base64 PNG data-URI for embedding in HTML cards.
    """
    import io, base64
    conn = get_db_connection()
    if conn is None:
        return None
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT DISTINCT t1_anchor_time, f1 FROM hashes WHERE song_name=?",
            (song_name,)
        )
        rows = cur.fetchall()
        if not rows:
            return None
        times = [r[0] for r in rows]
        freqs = [r[1] for r in rows]

        fig = plt.figure(figsize=(w_px/100, h_px/100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        fig.patch.set_facecolor('#0c140c')
        ax.set_facecolor('#0c140c')

        colors = ['#7dd3fc', '#fcd34d', '#f0abfc', '#86efac', '#fca5a5', '#a5b4fc']
        color = colors[abs(hash(song_name)) % len(colors)]

        ax.scatter(times, freqs, s=2.2, color=color, alpha=0.75, linewidths=0)
        ax.set_xlim(min(times), max(times) if max(times) > min(times) else min(times)+1)
        ax.set_ylim(0, 4000)
        ax.axis('off')

        buf = io.BytesIO()
        fig.savefig(buf, format='png', facecolor='#0c140c')
        plt.close(fig)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode()
        return f"data:image/png;base64,{b64}"
    except Exception:
        return None

# ═══════════════════════════════════════════
# IDENTIFY (unchanged logic)
# ═══════════════════════════════════════════
def identify_query_clip_sqlite(path, conn):
    from fingerprint import fingerprint_audio_file
    query_hashes, metadata = fingerprint_audio_file(path)
    cursor = conn.cursor()
    song_offset_matches = {}

    for hash_key, q_timestamps in query_hashes.items():
        f1, f2, dt = hash_key
        cursor.execute(
            "SELECT DISTINCT song_name, t1_anchor_time FROM hashes WHERE f1=? AND f2=? AND delta_t=?",
            (int(f1), int(f2), round(dt, 3))
        )
        for song_name, db_t1 in cursor.fetchall():
            song_offset_matches.setdefault(song_name, [])
            for q_t1 in q_timestamps:
                song_offset_matches[song_name].append(round(db_t1 - q_t1, 2))

    best_song, best_peak, best_offsets, match_counts, best_offset_val = \
        "Unknown / No Match", 0, [], {}, None

    for sname, offsets in song_offset_matches.items():
        if not offsets: continue
        match_counts[sname] = len(offsets)
        counts, edges = np.histogram(offsets, bins=np.arange(min(offsets)-1, max(offsets)+1, 0.5))
        peak = int(np.max(counts))
        if peak > best_peak:
            best_peak       = peak
            best_song       = sname
            best_offsets    = offsets
            best_offset_val = float(edges[int(np.argmax(counts))])

    return best_song, best_peak, best_offsets, match_counts, metadata, best_offset_val

# ═══════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════
def make_fig():
    fig, ax = plt.subplots(figsize=(11, 4))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#f8faf6')
    for sp in ax.spines.values(): sp.set_edgecolor('#e2e8dc')
    ax.tick_params(colors='#4b5563')
    ax.xaxis.label.set_color('#4b5563')
    ax.yaxis.label.set_color('#4b5563')
    ax.title.set_color('#0f1f0f')
    return fig, ax

def plot_spectrogram(metadata, title="Spectrogram"):
    fig, ax = make_fig()
    im = ax.imshow(
        metadata['spectrogram'], aspect='auto', origin='lower', cmap='YlOrRd',
        extent=[metadata['times'][0], metadata['times'][-1],
                metadata['frequencies'][0], metadata['frequencies'][-1]]
    )
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title, fontweight='bold', pad=10)
    cb = plt.colorbar(im, ax=ax)
    cb.set_label("dB", color='#4b5563')
    cb.ax.tick_params(colors='#4b5563')
    plt.tight_layout()
    return fig

def plot_constellation(metadata, title="Constellation Map"):
    fig, ax = make_fig()
    pt, pf = metadata['peak_times'], metadata['peak_freqs']
    if len(pt) > 0:
        ax.scatter(pt, pf, s=12, color='#2d5a27', alpha=0.7, edgecolors='#162316', linewidths=0.3)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"{title} — {len(pt)} peaks", fontweight='bold', pad=10)
    ax.grid(True, alpha=0.2, color='#e2e8dc')
    plt.tight_layout()
    return fig

def plot_histogram(offsets, song_name):
    fig, ax = make_fig()
    if offsets:
        ax.hist(offsets, bins=50, color='#2d5a27', edgecolor='#162316', alpha=0.8)
    ax.set_xlabel("Time Offset (s)"); ax.set_ylabel("Matching Hashes")
    ax.set_title(f"Offset Histogram — {song_name}", fontweight='bold', pad=10)
    ax.grid(True, alpha=0.2, color='#e2e8dc', axis='y')
    plt.tight_layout()
    return fig

# ═══════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════
if "mode" not in st.session_state:
    st.session_state.mode = "Single"

# ═══════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════
num_songs, unique_hashes, total_entries, db_size = get_db_stats()

uh_str = f"{unique_hashes/1000:.0f}K" if unique_hashes >= 1000 else str(unique_hashes)
te_str = f"{total_entries/1000:.0f}K" if total_entries >= 1000 else str(total_entries)

with st.sidebar:
    st.markdown("""
    <div class="sb-logo">
        <div class="sb-logo-icon">🎵</div>
        <div class="sb-logo-text">SongID</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<span class="sb-section">Database Stats</span>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="stats-grid">
        <div class="stat-cell">
            <span class="stat-num">{num_songs}</span>
            <span class="stat-lbl">Songs</span>
        </div>
        <div class="stat-cell">
            <span class="stat-num">{uh_str}</span>
            <span class="stat-lbl">Uniq Hashes</span>
        </div>
        <div class="stat-cell">
            <span class="stat-num">{te_str}</span>
            <span class="stat-lbl">Total Entries</span>
        </div>
        <div class="stat-cell">
            <span class="stat-num">{db_size:.1f}MB</span>
            <span class="stat-lbl">DB Size</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<span class="sb-section">Navigation</span>', unsafe_allow_html=True)

    s_active = "active" if st.session_state.mode == "Single" else ""
    b_active = "active" if st.session_state.mode == "Batch"  else ""
    d_active = "active" if st.session_state.mode == "Database" else ""
    st.markdown(f'<div class="sb-nav-item {s_active}">🎧 &nbsp; Single Clip</div>', unsafe_allow_html=True)
    if st.button("Open Single Clip", key="nav_s", use_container_width=True):
        st.session_state.mode = "Single"; st.rerun()

    st.markdown(f'<div class="sb-nav-item {b_active}">📂 &nbsp; Batch Mode</div>', unsafe_allow_html=True)
    if st.button("Open Batch Mode", key="nav_b", use_container_width=True):
        st.session_state.mode = "Batch"; st.rerun()

    st.markdown(f'<div class="sb-nav-item {d_active}">🗄️ &nbsp; Database</div>', unsafe_allow_html=True)
    if st.button("Open Database", key="nav_d", use_container_width=True):
        st.session_state.mode = "Database"; st.rerun()

# ═══════════════════════════════════════════
# DB CHECK
# ═══════════════════════════════════════════
db_conn = get_db_connection()
if db_conn is None:
    st.error("❌ music_database.db not found. Run build_database_sqlite.py first.")
    st.stop()

# ═══════════════════════════════════════════
# HEADER — dark green bg, white text
# ═══════════════════════════════════════════
st.markdown("""
<div class="page-header">
    <h1>🎵 Music Identifier</h1>
    <p>Identify songs using spectrogram fingerprinting — SQLite powered</p>
</div>
""", unsafe_allow_html=True)

# Mode cards
col1, col2, col3 = st.columns(3)
with col1:
    active = "active" if st.session_state.mode == "Single" else ""
    st.markdown(f'<div class="mode-card {active}"><h3>🎧 Single Clip</h3><p>Identify a single song using fast fingerprint matching.</p></div>', unsafe_allow_html=True)
    st.write("")
    if st.button("Activate Single Mode", use_container_width=True, key="ms"):
        st.session_state.mode = "Single"; st.rerun()

with col2:
    active = "active" if st.session_state.mode == "Batch" else ""
    st.markdown(f'<div class="mode-card {active}"><h3>📂 Batch Mode</h3><p>Process and catalog multiple audio files at once.</p></div>', unsafe_allow_html=True)
    st.write("")
    if st.button("Activate Batch Mode", use_container_width=True, key="mb"):
        st.session_state.mode = "Batch"; st.rerun()

with col3:
    active = "active" if st.session_state.mode == "Database" else ""
    st.markdown(f'<div class="mode-card {active}"><h3>🗄️ Database</h3><p>Browse constellation maps for every song in the database.</p></div>', unsafe_allow_html=True)
    st.write("")
    if st.button("Activate Database View", use_container_width=True, key="mdb"):
        st.session_state.mode = "Database"; st.rerun()

st.markdown("---")

# ═══════════════════════════════════════════
# SINGLE CLIP
# ═══════════════════════════════════════════
if st.session_state.mode == "Single":
    st.markdown('<span class="sec-hdr">Upload Audio Clip</span>', unsafe_allow_html=True)
    st.markdown('<span style="color: black;">Supports WAV, MP3, OGG, FLAC, M4A</span>', unsafe_allow_html=True)

    uploaded = st.file_uploader("", type=["wav","mp3","ogg","flac","m4a"], label_visibility="collapsed")

    if uploaded:
        st.audio(uploaded)

        temp_dir  = "./tmp"; os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, uploaded.name)
        with open(temp_path, "wb") as f: f.write(uploaded.getbuffer())

        run_key = f"res_{uploaded.name}_{uploaded.size}"
        if run_key not in st.session_state:
            with st.spinner("🔍 Analyzing audio..."):
                try:
                    st.session_state[run_key] = identify_query_clip_sqlite(temp_path, db_conn)
                except Exception as e:
                    st.error(f"Error during analysis: {e}"); st.stop()

        result = st.session_state.get(run_key)
        if not result: st.stop()
        best_song, best_score, best_offsets, match_counts, metadata, best_offset = result

        t_res, t_spec, t_con, t_hist = st.tabs(["🎯 Result","🎼 Spectrogram","⭐ Constellation","📈 Histogram"])

        with t_res:
            if best_song != "Unknown / No Match":
                pos = f"{best_offset:.1f}s" if best_offset is not None else "N/A"
                yt  = f"https://www.youtube.com/results?search_query={urllib.parse.quote(best_song)}"
                st.markdown(f"""
                <div class="result-card">
                    <span class="result-label">◈ IDENTIFIED SONG</span>
                    <span class="result-song">{best_song}</span>
                    <div class="result-meta">
                        Confidence: <b>{best_score}</b> &nbsp;|&nbsp;
                        Certainty: <b>{min(100,int((best_score/20)*100))}%</b>
                    </div>
                    <span class="result-time">📍 Matches at ~{pos} in reference song</span>
                    <a class="yt-btn" href="{yt}" target="_blank">▶ YouTube</a>
                </div>
                """, unsafe_allow_html=True)

                c1, c2, c3 = st.columns(3)
                c1.metric("Matched Song",     best_song[:18]+"…" if len(best_song)>18 else best_song)
                c2.metric("Confidence Score", best_score)
                c3.metric("Position in Song", pos)

                st.markdown('<span class="sec-hdr">Top Candidates</span>', unsafe_allow_html=True)
                for rank, (sname, score) in enumerate(sorted(match_counts.items(), key=lambda x: x[1], reverse=True)[:5], 1):
                    st.markdown(f"""
                    <div class="cand-row">
                        <span class="cand-rank">#{rank}</span>
                        <span class="cand-name">{sname}</span>
                        <span class="cand-score">{score} matches</span>
                    </div>""", unsafe_allow_html=True)
            else:
                st.warning("⚠️ No match found in database.")

        with t_spec:
            try: st.pyplot(plot_spectrogram(metadata, f"Spectrogram — {uploaded.name}"))
            except Exception as e: st.error(f"Plot error: {e}")

        with t_con:
            try: st.pyplot(plot_constellation(metadata, f"Constellation — {uploaded.name}"))
            except Exception as e: st.error(f"Plot error: {e}")

        with t_hist:
            if best_offsets:
                try: st.pyplot(plot_histogram(best_offsets, best_song))
                except Exception as e: st.error(f"Plot error: {e}")
            else:
                st.info("No offset data available.")

        try: os.remove(temp_path)
        except: pass

# ═══════════════════════════════════════════
# BATCH MODE
# ═══════════════════════════════════════════
elif st.session_state.mode == "Batch":
    st.markdown('<span class="sec-hdr">📂 Batch Processing</span>', unsafe_allow_html=True)
    st.write("Upload multiple audio files — WAV, MP3, OGG, FLAC, M4A")

    uploaded_files = st.file_uploader(
        "Upload files", type=["wav","mp3","ogg","flac","m4a"],
        accept_multiple_files=True, label_visibility="collapsed"
    )

    if uploaded_files:
        st.info(f"📁 {len(uploaded_files)} files selected")

        if st.button("▶️ Process All Files", use_container_width=True, key="process_batch"):
            results, batch_data = [], []
            prog   = st.progress(0)
            status = st.empty()
            temp_dir = "./tmp"; os.makedirs(temp_dir, exist_ok=True)

            for idx, uf in enumerate(uploaded_files):
                status.write(f"Processing **{idx+1}/{len(uploaded_files)}**: {uf.name}")
                prog.progress((idx+1)/len(uploaded_files))

                temp_path = os.path.join(temp_dir, uf.name)
                with open(temp_path, "wb") as f: f.write(uf.getbuffer())

                try:
                    best_song, best_score, best_offsets, match_counts, metadata, best_offset = \
                        identify_query_clip_sqlite(temp_path, db_conn)
                    matched    = best_song != "Unknown / No Match"
                    pred_clean = os.path.splitext(best_song if matched else "UNKNOWN")[0]
                    pos        = f"{best_offset:.1f}s" if best_offset is not None else "N/A"

                    results.append({"filename": os.path.splitext(uf.name)[0], "prediction": pred_clean})
                    batch_data.append({
                        "filename": uf.name, "song": best_song,
                        "matched": matched, "score": best_score,
                        "position": pos, "offsets": best_offsets,
                        "metadata": metadata
                    })
                except Exception as e:
                    results.append({"filename": os.path.splitext(uf.name)[0], "prediction": "ERROR"})
                    batch_data.append({
                        "filename": uf.name, "song": f"ERROR: {e}",
                        "matched": False, "score": 0,
                        "position": "N/A", "offsets": [], "metadata": None
                    })

                try: os.remove(temp_path)
                except: pass

            prog.empty(); status.empty()
            st.success("✅ Batch processing complete!")

            # Summary
            identified = sum(1 for r in results if r['prediction'] not in ['UNKNOWN','ERROR'])
            errors     = sum(1 for r in results if r['prediction'] == 'ERROR')
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Files",  len(results))
            c2.metric("Identified",   identified)
            c3.metric("Errors",       errors)

            # Table
            st.markdown('<span class="sec-hdr">Results Table</span>', unsafe_allow_html=True)
            df_res = pd.DataFrame(results)
            st.dataframe(df_res, use_container_width=True)
            st.download_button(
                "📥 Download results.csv",
                data=df_res.to_csv(index=False),
                file_name="results.csv", mime="text/csv"
            )

            # Per-file analysis — using columns instead of expander+tabs to avoid rendering issues
            st.markdown('<span class="sec-hdr">Per-File Analysis</span>', unsafe_allow_html=True)

            for item in batch_data:
                icon = "✅" if item['matched'] else "❌"
                with st.expander(f"{icon}  {item['filename']}  →  {item['song']}", expanded=False):

                    # Match info row
                    if item['matched']:
                        yt = f"https://www.youtube.com/results?search_query={urllib.parse.quote(item['song'])}"
                        st.markdown(f"""
                        <div class="batch-card">
                            <div class="batch-fname">{item['filename']}</div>
                            <div class="batch-sname">🎵 {item['song']}</div>
                            <span class="batch-pos">📍 Matches at ~{item['position']} in reference song</span>
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown(f'<a class="yt-btn" style="margin-left:0;" href="{yt}" target="_blank">▶ Watch on YouTube</a><br><br>', unsafe_allow_html=True)
                    else:
                        st.warning(f"No match found: {item['song']}")

                    # Visualizations — render directly (no nested tabs to avoid rendering bugs)
                    if item['metadata'] is not None:
                        st.markdown('<span class="batch-subhdr">🎼 Spectrogram</span>', unsafe_allow_html=True)
                        try:
                            st.pyplot(plot_spectrogram(item['metadata'], f"Spectrogram — {item['filename']}"))
                        except Exception as e:
                            st.error(f"Spectrogram error: {e}")

                        st.markdown('<span class="batch-subhdr">⭐ Constellation Map</span>', unsafe_allow_html=True)
                        try:
                            st.pyplot(plot_constellation(item['metadata'], f"Constellation — {item['filename']}"))
                        except Exception as e:
                            st.error(f"Constellation error: {e}")

                        st.markdown('<span class="batch-subhdr">📈 Offset Histogram</span>', unsafe_allow_html=True)
                        try:
                            if item['offsets']:
                                st.pyplot(plot_histogram(item['offsets'], item['song']))
                            else:
                                st.info("No offset data available.")
                        except Exception as e:
                            st.error(f"Histogram error: {e}")

# ═══════════════════════════════════════════
# DATABASE VIEW — constellation maps for every song
# ═══════════════════════════════════════════
else:
    st.markdown('<span class="sec-hdr">🗄️ In The Database</span>', unsafe_allow_html=True)

    song_rows = get_song_list_with_counts()

    if not song_rows:
        st.warning("⚠️ No songs found in the database.")
    else:
        search = st.text_input(
            "Search songs", placeholder="🔍 Search songs…", label_visibility="collapsed"
        )
        if search:
            filtered = [r for r in song_rows if search.lower() in r[0].lower()]
        else:
            filtered = song_rows

        st.markdown(
            f'<span class="db-grid-label">{len(filtered)} of {len(song_rows)} songs</span>',
            unsafe_allow_html=True
        )

        if not filtered:
            st.info("No songs match your search.")
        else:
            cols_per_row = 4
            with st.spinner("🎨 Rendering constellation maps..."):
                for i in range(0, len(filtered), cols_per_row):
                    row_songs = filtered[i:i + cols_per_row]
                    cols = st.columns(cols_per_row)
                    for col, (song_name, hash_count) in zip(cols, row_songs):
                        with col:
                            thumb = get_constellation_thumb(song_name)
                            img_html = (
                                f'<img src="{thumb}" />'
                                if thumb else
                                '<div style="height:120px;display:flex;align-items:center;'
                                'justify-content:center;color:#7d9a7d;font-size:0.75rem;">'
                                'No data</div>'
                            )
                            st.markdown(f"""
                            <div class="db-card">
                                {img_html}
                                <span class="db-card-name">{song_name}</span>
                                <span class="db-card-hashes">{hash_count:,} hashes</span>
                            </div>
                            """, unsafe_allow_html=True)
                            st.write("")