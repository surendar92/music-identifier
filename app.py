# app.py (Enhanced UI Version)
# Streamlit app with SQLite database backend + improved UI
# Run with: streamlit run app.py

import streamlit as st
import sqlite3
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from fingerprint import fingerprint_audio_file


# ============================================================================
# LOAD SONG METADATA
# ============================================================================
@st.cache_resource
def load_song_metadata():
    """Load artist names and YouTube URLs"""
    try:
        with open("song_metadata.json", "r") as f:
            data = json.load(f)
            return data.get("song_metadata", {})
    except:
        return {}


SONG_METADATA = load_song_metadata()

# ============================================================================
# CUSTOM STYLING
# ============================================================================
st.markdown("""
    <style>
        /* Dark theme colors */
        :root {
            --primary: #667eea;
            --secondary: #764ba2;
            --accent: #00D1B2;
            --bg-dark: #0F1419;
            --bg-card: #1E2235;
            --border: #2D3142;
            --text-primary: #FFFFFF;
            --text-secondary: #A3A8B4;
        }

        /* Button styling */
        div.stButton > button:first-child {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 8px;
            border: none;
            padding: 10px 24px;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        div.stButton > button:first-child:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(118, 75, 162, 0.4);
        }

        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 10px 20px;
            border-radius: 8px;
        }

        /* Graph background */
        .stPlotlyChart, .stPyplotChart {
            background: transparent;
        }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="🎵 Music Identifier",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# DATABASE CONNECTION (Cached)
# ============================================================================
@st.cache_resource
def get_db_connection():
    """Connect to SQLite database"""
    try:
        conn = sqlite3.connect("music_database.db", check_same_thread=False)
        return conn
    except Exception as e:
        st.error(f"❌ Could not connect to database: {e}")
        return None


@st.cache_resource
def get_db_stats():
    """Get database statistics"""
    conn = get_db_connection()
    if conn is None:
        return None, None, None

    cursor = conn.cursor()
    cursor.execute("""
        SELECT COUNT(*) FROM (
            SELECT DISTINCT f1, f2, delta_t FROM hashes
        )
    """)
    unique_hashes = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT song_name) FROM hashes")
    num_songs = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM hashes")
    total_entries = cursor.fetchone()[0]

    return num_songs, unique_hashes, total_entries


# ============================================================================
# HELPER: Format song name for display
# ============================================================================
def format_song_name(song_name):
    """
    Convert database name to display name.
    In this database, song_name already contains spaces (e.g. "Hey Jude").
    The underscore character is only used as a stand-in for an apostrophe
    (e.g. "Don_t Stop Me Now" -> "Don't Stop Me Now").
    """
    return song_name.replace("_", "'")


def get_song_info(song_name):
    """Get artist and YouTube URL for a song"""
    return SONG_METADATA.get(song_name, {
        "artist": "Unknown Artist",
        "youtube_url": None
    })


# ============================================================================
# IDENTIFY FUNCTION (Using SQLite) - UNCHANGED LOGIC
# ============================================================================
def identify_query_clip_sqlite(query_audio_path, conn):
    """
    Match query against SQLite database using YOUR identify logic
    But with fast SQL lookups instead of in-memory dictionary
    """
    print(f"Fingerprinting query: {query_audio_path.split('/')[-1]}...")
    query_hashes, metadata = fingerprint_audio_file(query_audio_path)

    cursor = conn.cursor()
    song_offset_matches = {}

    # 1. For each hash in the query, look it up in the database
    for hash_key, q_timestamps in query_hashes.items():
        f1, f2, dt = hash_key

        # Fast indexed lookup in SQLite
        cursor.execute("""
            SELECT DISTINCT song_name, t1_anchor_time 
            FROM hashes 
            WHERE f1 = ? AND f2 = ? AND delta_t = ?
        """, (int(f1), int(f2), round(dt, 3)))

        results = cursor.fetchall()

        # For each match, calculate offset
        for song_name, db_t1 in results:
            if song_name not in song_offset_matches:
                song_offset_matches[song_name] = []

            for q_t1 in q_timestamps:
                offset = db_t1 - q_t1
                song_offset_matches[song_name].append(round(offset, 2))

    # 2. Find song with highest consensus
    best_song = "Unknown / No Match"
    highest_peak_count = 0
    best_offsets_list = []
    match_counts = {}

    for song_name, offsets in song_offset_matches.items():
        if len(offsets) == 0:
            continue

        match_counts[song_name] = len(offsets)

        # Group offsets to find histogram peak
        counts, bin_edges = np.histogram(
            offsets,
            bins=np.arange(min(offsets) - 1, max(offsets) + 1, 0.5)
        )
        max_spike = np.max(counts)

        if max_spike > highest_peak_count:
            highest_peak_count = max_spike
            best_song = song_name
            best_offsets_list = offsets

    print(f"Prediction: '{best_song}' with consensus peak score of {highest_peak_count}!")

    return best_song, highest_peak_count, best_offsets_list, match_counts, metadata


# ============================================================================
# MAIN UI
# ============================================================================
st.title("🎵 Music Identifier – Zapptain America")
st.markdown("**Identify songs using spectrogram fingerprinting** powered by SQLite")
st.markdown("---")

# Get database connection
db_conn = get_db_connection()
if db_conn is None:
    st.error("❌ music_database.db not found!")
    st.stop()

# Show database stats in sidebar
with st.sidebar:
    st.markdown("### 📊 Database Stats")
    num_songs, unique_hashes, total_entries = get_db_stats()
    if num_songs:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🎵 Songs", num_songs)
            st.metric("🔐 Hashes", f"{unique_hashes:,}")
        with col2:
            st.metric("📦 Entries", f"{total_entries:,}")

            db_size = os.path.getsize("music_database.db") / (1024 * 1024)
            st.metric("💾 DB Size", f"{db_size:.2f} MB")

# ============================================================================
# MODE SELECTION
# ============================================================================
if "mode" not in st.session_state:
    st.session_state.mode = "Single"

col1, col2 = st.columns(2)

with col1:
    is_single = st.session_state.mode == "Single"

    st.markdown(
        f"""
        <div style="
            padding: 20px; 
            border-radius: 10px; 
            border: 2px solid {'#00D1B2' if is_single else '#4F5366'}; 
            background-color: {'#1E2235' if is_single else 'transparent'};
            text-align: center;
        ">
            <h3>🎧 Single Clip</h3>
            <p style="color: #A3A8B4; font-size: 14px;">Identify a single song from an audio clip.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    if st.button("Activate Single Mode", use_container_width=True, type="primary" if is_single else "secondary"):
        st.session_state.mode = "Single"
        st.rerun()

with col2:
    is_batch = st.session_state.mode == "Batch"

    st.markdown(
        f"""
        <div style="
            padding: 20px; 
            border-radius: 10px; 
            border: 2px solid {'#FF4B4B' if is_batch else '#4F5366'}; 
            background-color: {'#1E2235' if is_batch else 'transparent'};
            text-align: center;
        ">
            <h3>📂 Batch Mode</h3>
            <p style="color: #A3A8B4; font-size: 14px;">Process multiple audio files at once.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    if st.button("Activate Batch Mode", use_container_width=True, type="primary" if is_batch else "secondary"):
        st.session_state.mode = "Batch"
        st.rerun()

st.markdown("---")

# ============================================================================
# MODE 1: SINGLE CLIP
# ============================================================================
if st.session_state.mode == "Single":
    st.subheader("🎧 Upload an audio clip to identify")
    st.markdown("*Supports: WAV, MP3, OGG, FLAC, M4A*")

    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=["wav", "mp3", "ogg", "flac", "m4a"]
    )

    if uploaded_file:
        st.success(f"✓ File uploaded: **{uploaded_file.name}**")

        # Save uploaded file temporarily
        temp_dir = "./tmp"
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Identify the song
        with st.spinner("🔍 Analyzing audio..."):
            best_song, best_score, best_offsets_list, match_counts, metadata = identify_query_clip_sqlite(
                temp_path, db_conn
            )
            st.session_state.metadata = metadata
            st.session_state.best_song = best_song
            st.session_state.best_score = best_score
            st.session_state.best_offsets = best_offsets_list
            st.session_state.match_counts = match_counts

        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(
            ["📊 Result & Info", "🎼 Spectrogram", "⭐ Constellation", "📈 Histogram"]
        )

        # ==============================================================================
        # TAB 1: RESULT WITH ARTIST & YOUTUBE
        # ==============================================================================
        with tab1:
            if best_song != "Unknown / No Match":
                song_info = get_song_info(best_song)
                display_name = format_song_name(best_song)
                artist = song_info.get("artist", "Unknown Artist")
                youtube_url = song_info.get("youtube_url")

                # Main match banner
                st.markdown(
                    f"""
                    <div style="
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 28px;
                        border-radius: 12px;
                        border: 1px solid #2D3142;
                        margin-bottom: 24px;
                    ">
                        <span style="color: #FFFFFF; font-weight: 700; font-size: 12px; uppercase;">🎉 MATCH IDENTIFIED</span>
                        <h1 style="margin: 8px 0 0 0; color: #FFFFFF; font-size: 42px;">{display_name}</h1>
                        <p style="margin: 6px 0 0 0; color: #E0E0E0; font-size: 18px;">🎤 {artist}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Metrics
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("🎯 Confidence Score", f"{best_score:,}")

                with col2:
                    confidence_pct = min(100, int((best_score / 20) * 100))
                    st.metric("📊 Match Certainty", f"{confidence_pct}%")

                with col3:
                    st.metric("📍 Hash Matches", len(best_offsets_list))

                # YouTube Link
                if youtube_url:
                    st.markdown(f"""
                    <div style="text-align: center; margin: 20px 0;">
                        <a href="{youtube_url}" target="_blank" style="
                            display: inline-block;
                            background: linear-gradient(135deg, #FF4B4B 0%, #FF6B6B 100%);
                            color: white;
                            padding: 12px 32px;
                            border-radius: 8px;
                            text-decoration: none;
                            font-weight: bold;
                            font-size: 16px;
                            transition: transform 0.2s;
                        ">
                            ▶️ Watch on YouTube
                        </a>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("### 📋 Top Candidates")
                top_candidates = sorted(match_counts.items(), key=lambda x: x[1], reverse=True)[:5]

                candidates_html = """
                <table style="width:100%; border-collapse: collapse; margin-top:10px; font-size:14px; color:#E2E8F0;">
                    <thead>
                        <tr style="border-bottom: 2px solid #2D3142;">
                            <th style="padding: 10px; text-align: left; color:#A3A8B4;">Rank</th>
                            <th style="padding: 10px; text-align: left; color:#A3A8B4;">Song</th>
                            <th style="padding: 10px; text-align: right; color:#A3A8B4;">Matches</th>
                        </tr>
                    </thead>
                    <tbody>
                """

                for rank, (song_name, score) in enumerate(top_candidates, 1):
                    display_song = format_song_name(song_name)
                    bg_color = "rgba(102, 126, 234, 0.1)" if rank == 1 else "transparent"
                    text_color = "#667eea" if rank == 1 else "#E2E8F0"
                    medal = "🥇" if rank == 1 else f"#{rank}"

                    candidates_html += f"""
                    <tr style="border-bottom: 1px solid #1E2235; background-color: {bg_color};">
                        <td style="padding: 12px; font-weight: 600; color:{text_color};">{medal}</td>
                        <td style="padding: 12px;">{display_song}</td>
                        <td style="padding: 12px; text-align: right; font-weight: 600;">{score:,}</td>
                    </tr>
                    """

                candidates_html += """
                    </tbody>
                </table>
                """

                st.markdown(candidates_html, unsafe_allow_html=True)

            else:
                st.warning("⚠️ No match found in database.")

        # ==============================================================================
        # TAB 2: SPECTROGRAM (IMPROVED LAYOUT)
        # ==============================================================================
        with tab2:
            st.subheader("🎼 Spectrogram Visualization")
            st.markdown("Time-frequency representation of the audio signal")

            if "metadata" in st.session_state and st.session_state.metadata is not None:
                try:
                    current_metadata = st.session_state.metadata
                    spec_db = current_metadata['spectrogram']
                    times = current_metadata['times']
                    freqs = current_metadata['frequencies']

                    # Create larger, better-styled figure
                    fig, ax = plt.subplots(figsize=(14, 7))
                    fig.patch.set_facecolor('#0F1419')
                    ax.set_facecolor('#1E2235')

                    im = ax.imshow(
                        spec_db,
                        aspect='auto',
                        origin='lower',
                        cmap='magma',
                        extent=[times[0], times[-1], freqs[0], freqs[-1]]
                    )

                    ax.set_xlabel("Time (seconds)", fontsize=12, color='#A3A8B4')
                    ax.set_ylabel("Frequency (Hz)", fontsize=12, color='#A3A8B4')
                    ax.set_title("Spectrogram (dB scale)", fontsize=14, color='#FFFFFF', pad=20)

                    # Style colorbar
                    cbar = plt.colorbar(im, ax=ax, label="Magnitude (dB)")
                    cbar.ax.tick_params(colors='#A3A8B4')
                    cbar.ax.yaxis.label.set_color('#A3A8B4')

                    # Style ticks
                    ax.tick_params(colors='#A3A8B4', labelsize=10)
                    ax.spines['bottom'].set_color('#2D3142')
                    ax.spines['left'].set_color('#2D3142')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)

                    st.pyplot(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.info("Run identification first")

        # ==============================================================================
        # TAB 3: CONSTELLATION (IMPROVED LAYOUT)
        # ==============================================================================
        with tab3:
            st.subheader("⭐ Peak Constellation")
            st.markdown("Spectral peaks extracted from the audio signal")

            if "metadata" in st.session_state and st.session_state.metadata is not None:
                try:
                    current_metadata = st.session_state.metadata
                    peak_times = current_metadata['peak_times']
                    peak_freqs = current_metadata['peak_freqs']

                    if len(peak_times) > 0:
                        fig, ax = plt.subplots(figsize=(14, 7))
                        fig.patch.set_facecolor('#0F1419')
                        ax.set_facecolor('#1E2235')

                        ax.scatter(
                            peak_times,
                            peak_freqs,
                            alpha=0.7,
                            s=60,
                            color='#FF9500',
                            edgecolors='#FFB84D',
                            linewidth=1
                        )

                        ax.set_xlabel("Time (seconds)", fontsize=12, color='#A3A8B4')
                        ax.set_ylabel("Frequency (Hz)", fontsize=12, color='#A3A8B4')
                        ax.set_title(f"Peak Constellation – {len(peak_times)} peaks detected", fontsize=14,
                                     color='#FFFFFF', pad=20)
                        ax.grid(visible=True, alpha=0.2, color='#2D3142')

                        ax.tick_params(colors='#A3A8B4', labelsize=10)
                        ax.spines['bottom'].set_color('#2D3142')
                        ax.spines['left'].set_color('#2D3142')
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)

                        st.pyplot(fig, use_container_width=True)
                    else:
                        st.info("No peaks detected")

                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.info("Run identification first")

        # ==============================================================================
        # TAB 4: OFFSET HISTOGRAM (IMPROVED LAYOUT)
        # ==============================================================================
        with tab4:
            st.subheader("📈 Offset Histogram")
            st.markdown("Voting histogram showing where hashes align")

            if "best_offsets" in st.session_state and st.session_state.best_offsets:
                try:
                    best_offsets = st.session_state.best_offsets
                    best_song = st.session_state.best_song

                    fig, ax = plt.subplots(figsize=(14, 7))
                    fig.patch.set_facecolor('#0F1419')
                    ax.set_facecolor('#1E2235')

                    ax.hist(
                        best_offsets,
                        bins=50,
                        color='#667eea',
                        edgecolor='#764ba2',
                        alpha=0.8,
                        linewidth=1.5
                    )

                    ax.set_xlabel("Time Offset (seconds)", fontsize=12, color='#A3A8B4')
                    ax.set_ylabel("Matching Hashes", fontsize=12, color='#A3A8B4')
                    ax.set_title(f"Offset Histogram – {format_song_name(best_song)}", fontsize=14, color='#FFFFFF',
                                 pad=20)
                    ax.grid(True, alpha=0.2, axis='y', color='#2D3142')

                    ax.tick_params(colors='#A3A8B4', labelsize=10)
                    ax.spines['bottom'].set_color('#2D3142')
                    ax.spines['left'].set_color('#2D3142')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)

                    st.pyplot(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.info("Run identification first")

        # Cleanup
        try:
            os.remove(temp_path)
        except:
            pass

# ============================================================================
# MODE 2: BATCH MODE
# ============================================================================
else:  # Batch mode
    st.subheader("📁 Batch Processing & Analytics")

    uploaded_files = st.file_uploader(
        "Choose multiple audio files",
        type=["wav", "mp3", "ogg", "flac", "m4a"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.info(f"📁 {len(uploaded_files)} files selected")

        if st.button("▶️ Process All Files", type="primary", use_container_width=True):
            results = []
            progress_bar = st.progress(0)
            status_placeholder = st.empty()

            for idx, uploaded_file in enumerate(uploaded_files):
                status_placeholder.text(
                    f"⏳ Processing {idx + 1}/{len(uploaded_files)}: {uploaded_file.name}"
                )
                progress_bar.progress((idx + 1) / len(uploaded_files))

                temp_dir = "./tmp"
                os.makedirs(temp_dir, exist_ok=True)
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                try:
                    best_song, _, _, _, _ = identify_query_clip_sqlite(temp_path, db_conn)
                    filename_without_ext = os.path.splitext(uploaded_file.name)[0]
                    prediction = best_song if best_song != "Unknown / No Match" else "UNKNOWN"

                    results.append({
                        "filename": filename_without_ext,
                        "prediction": format_song_name(prediction)
                    })

                except Exception as e:
                    filename_without_ext = os.path.splitext(uploaded_file.name)[0]
                    results.append({
                        "filename": filename_without_ext,
                        "prediction": "ERROR"
                    })

                try:
                    os.remove(temp_path)
                except:
                    pass

            status_placeholder.empty()
            progress_bar.empty()

            st.success("✅ Batch processing complete!")

            # Display results
            df_results = pd.DataFrame(results)
            st.dataframe(df_results, use_container_width=True)

            csv_content = df_results.to_csv(index=False)
            st.download_button(
                label="📥 Download results.csv",
                data=csv_content,
                file_name="results.csv",
                mime="text/csv"
            )

            # Summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Files", len(results))
            with col2:
                unknowns = sum(1 for r in results if r['prediction'] in ['UNKNOWN', 'ERROR'])
                st.metric("Identified", len(results) - unknowns)
            with col3:
                errors = sum(1 for r in results if r['prediction'] == 'ERROR')
                st.metric("Errors", errors)

# ============================================================================
# SIDEBAR INFO
# ============================================================================
st.sidebar.markdown("---")
st.sidebar.markdown("### 📖 About")
st.sidebar.markdown("""
**Music Identifier** uses spectrogram fingerprinting:
- Fast indexed SQLite queries
- High accuracy matching
- Real-time identification
- Batch processing support

**Course:** EE200 – Signals, Systems, Networks
**Project:** Q3B – Music Identification
""")