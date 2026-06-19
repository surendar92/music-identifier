# app.py (SQLite version)
# Streamlit app with SQLite database backend
# Run with: streamlit run app.py

import streamlit as st
import sqlite3
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from fingerprint import fingerprint_audio_file

st.markdown("""
    <style>
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
        .sidebar-header {
            font-size: 20px;
            font-weight: bold;
            color: #764ba2;
            margin-bottom: 15px;
        }
        .stat-card {
            background-color: #1E2235;
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 10px;
            border-left: 4px solid #667eea;
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
# IDENTIFY FUNCTION (Using SQLite)
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
st.markdown("**Identify songs using spectrogram fingerprinting** (SQLite-powered)")
st.markdown("---")

# Get database connection
db_conn = get_db_connection()
if db_conn is None:
    st.error("❌ music_database.db not found!")
    st.info("""
    Before running:
    1. Create a `songs/` folder with reference songs
    2. Run: `python build_database_sqlite.py`
    3. This creates `music_database.db`
    4. Then run: `streamlit run app.py`
    """)
    st.stop()

# Show database stats in refined sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-header">📊 Database Dashboard</div>', unsafe_allow_html=True)
    num_songs, unique_hashes, total_entries = get_db_stats()
    if num_songs:
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        st.metric("Total Reference Songs", num_songs)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        st.metric("Unique Fingerprint Hashes", f"{unique_hashes:,}")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        st.metric("Total Structural Entries", f"{total_entries:,}")
        st.markdown('</div>', unsafe_allow_html=True)

        db_size = os.path.getsize("music_database.db") / (1024 * 1024)
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        st.metric("Database Storage Size", f"{db_size:.2f} MB")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔍 Reference Browser")
    cursor = db_conn.cursor()
    cursor.execute("SELECT DISTINCT song_name FROM hashes ORDER BY song_name")
    avail_songs = [row[0] for row in cursor.fetchall()]

    if avail_songs:
        selected_db_song = st.selectbox("View Constellation from DB", avail_songs)
        if selected_db_song:
            # Modified query to pull unique coordinates matching individual song footprints across the full timeline
            cursor.execute(
                "SELECT DISTINCT t1_anchor_time, f1 FROM hashes WHERE song_name = ?",
                (selected_db_song,)
            )
            points = cursor.fetchall()
            if points:
                pt_times, pt_freqs = zip(*points)
                fig_side, ax_side = plt.subplots(figsize=(4, 3))
                ax_side.scatter(pt_times, pt_freqs, s=4, color='orange', alpha=0.6, edgecolors='none')
                ax_side.set_title(f"Constellation Map: {selected_db_song}", fontsize=9)
                ax_side.set_xlabel("Time (seconds)", fontsize=7)
                ax_side.set_ylabel("Frequency (Hz)", fontsize=7)
                ax_side.tick_params(axis='both', which='major', labelsize=6)
                ax_side.grid(True, alpha=0.2)
                st.pyplot(fig_side)

# ==============================================================================
# MODERN MODE SELECTION (MAIN PAGE)
# ==============================================================================

st.write("### 🛠️ Select Operation Mode")

# Initialize session state for mode if it doesn't exist
if "mode" not in st.session_state:
    st.session_state.mode = "Single"

# Create two clean columns for the mode selection cards
col1, col2 = st.columns(2)

with col1:
    # Card styling for Single Clip mode
    is_single = st.session_state.mode == "Single"

    # Visual indicator using a nice markdown block
    st.markdown(
        f"""
        <div style="
            padding: 20px; 
            border-radius: 10px; 
            border: 2px solid {'#FF4B4B' if is_single else '#4F5366'}; 
            background-color: {'#1E2235' if is_single else 'transparent'};
            text-align: center;
        ">
            <h3>🎧 Single Clip Mode</h3>
            <p style="color: #A3A8B4; font-size: 14px;">Identify a single song with fully detailed execution steps.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write("")  # Spacer
    if st.button("Activate Single Mode", use_container_width=True, type="primary" if is_single else "secondary"):
        st.session_state.mode = "Single"
        st.rerun()

with col2:
    # Card styling for Batch Mode
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
            <h3>📂 Batch Operations Mode</h3>
            <p style="color: #A3A8B4; font-size: 14px;">Process multiple files simultaneously and audit tracking metrics.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write("")  # Spacer
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

    col1, col2 = st.columns([2, 1])

    with col1:
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

        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(
            ["📊 Result", "🎼 Spectrogram", "⭐ Constellation Peaks", "📈 Offset Histogram"]
        )

        # ===== TAB 1: IDENTIFICATION RESULT =====
        with tab1:
            st.subheader("🎯 Identification Result")

            with st.spinner("🔍 Analyzing audio (querying SQLite database)..."):
                try:
                    # SQLite-based identification
                    best_song, best_score, best_offsets_list, match_counts, metadata = identify_query_clip_sqlite(
                        temp_path,
                        db_conn
                    )

                    if best_song != "Unknown / No Match":
                        # Display result
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric("🎵 Matched Song", best_song)

                        with col2:
                            st.metric("🎯 Confidence", best_score)

                        with col3:
                            if best_score > 0:
                                confidence_pct = min(100, (best_score / 20) * 100)
                                st.metric("📊 Certainty", f"{confidence_pct:.0f}%")

                        st.success(f"✅ Song identified as **{best_song}**")

                        # Top candidates
                        st.markdown("### Top 5 Candidates")
                        top_candidates = sorted(match_counts.items(), key=lambda x: x[1], reverse=True)[:5]

                        for rank, (song_name, score) in enumerate(top_candidates, 1):
                            col_rank, col_name, col_score = st.columns([0.5, 2, 1])
                            with col_rank:
                                st.write(f"**#{rank}**")
                            with col_name:
                                st.write(song_name)
                            with col_score:
                                st.write(f"{score} matches")

                    else:
                        st.warning("⚠️ No match found in database.")

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    import traceback

                    st.write(traceback.format_exc())

        # ===== TAB 2: SPECTROGRAM =====
        with tab2:
            st.subheader("🎼 Spectrogram Visualization")
            try:
                spec_db = metadata['spectrogram']
                times = metadata['times']
                freqs = metadata['frequencies']

                fig, ax = plt.subplots(figsize=(12, 5))
                im = ax.imshow(
                    spec_db,
                    aspect='auto',
                    origin='lower',
                    cmap='magma',
                    extent=[times[0], times[-1], freqs[0], freqs[-1]]
                )
                ax.set_xlabel("Time (seconds)")
                ax.set_ylabel("Frequency (Hz)")
                ax.set_title("Spectrogram (dB scale)")
                plt.colorbar(im, ax=ax, label="Magnitude (dB)")
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Error: {e}")

        # ===== TAB 3: CONSTELLATION =====
        with tab3:
            st.subheader("⭐ Peak Constellation")
            try:
                peak_times = metadata['peak_times']
                peak_freqs = metadata['peak_freqs']

                if len(peak_times) > 0:
                    fig, ax = plt.subplots(figsize=(12, 5))
                    ax.scatter(peak_times, peak_freqs, alpha=0.6, s=40, color='orange', edgecolors='darkorange')
                    ax.set_xlabel("Time (seconds)")
                    ax.set_ylabel("Frequency (Hz)")
                    ax.set_title(f"Peak Constellation – {len(peak_times)} peaks")
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                else:
                    st.info("No peaks detected")

            except Exception as e:
                st.error(f"Error: {e}")

        # ===== TAB 4: OFFSET HISTOGRAM =====
        with tab4:
            st.subheader("📈 Offset Histogram")
            try:
                if best_offsets_list and len(best_offsets_list) > 0:
                    fig, ax = plt.subplots(figsize=(12, 5))
                    ax.hist(best_offsets_list, bins=50, color='steelblue', edgecolor='navy', alpha=0.7)
                    ax.set_xlabel("Time Offset (seconds)")
                    ax.set_ylabel("Matching Hashes")
                    ax.set_title(f"Offset Histogram for '{best_song}'")
                    ax.grid(True, alpha=0.3, axis='y')
                    st.pyplot(fig)
                else:
                    st.info("No offset data")

            except Exception as e:
                st.error(f"Error: {e}")

        # Cleanup
        try:
            os.remove(temp_path)
        except:
            pass

# ============================================================================
# MODE 2: BATCH MODE
# ============================================================================
else:  # Batch mode
    st.subheader("📁 Batch Processing & Computational Breakdown")

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
                    best_song, best_score, best_offsets_list, match_counts, metadata = identify_query_clip_sqlite(
                        temp_path, db_conn
                    )
                    filename_without_ext = os.path.splitext(uploaded_file.name)[0]
                    prediction = best_song if best_song != "Unknown / No Match" else "UNKNOWN"

                    results.append({
                        "filename": filename_without_ext,
                        "prediction": prediction
                    })

                    # Dynamic analytical visualization rendering for each evaluated batch file
                    with st.expander(f"📊 Signal Analysis Breakdown: {uploaded_file.name}", expanded=False):
                        st.markdown(
                            f"**Prediction Matched:** `{prediction}` | **Consensus Peak Score:** `{best_score}`")

                        col_p1, col_p2, col_p3 = st.columns(3)

                        with col_p1:
                            st.write("🎼 **Spectrogram Plot**")
                            try:
                                fig_b1, ax_b1 = plt.subplots(figsize=(4, 2.5))
                                ax_b1.imshow(
                                    metadata['spectrogram'], aspect='auto', origin='lower', cmap='magma',
                                    extent=[metadata['times'][0], metadata['times'][-1], metadata['frequencies'][0],
                                            metadata['frequencies'][-1]]
                                )
                                ax_b1.set_xlabel("Time (s)", fontsize=6)
                                ax_b1.set_ylabel("Freq (Hz)", fontsize=6)
                                ax_b1.tick_params(labelsize=6)
                                st.pyplot(fig_b1)
                            except Exception as ex:
                                st.error(f"Render Error: {ex}")

                        with col_p2:
                            st.write("⭐ **Peak Constellation**")
                            try:
                                fig_b2, ax_b2 = plt.subplots(figsize=(4, 2.5))
                                ax_b2.scatter(metadata['peak_times'], metadata['peak_freqs'], alpha=0.6, s=8,
                                              color='orange')
                                ax_b2.set_xlabel("Time (s)", fontsize=6)
                                ax_b2.set_ylabel("Freq (Hz)", fontsize=6)
                                ax_b2.tick_params(labelsize=6)
                                ax_b2.grid(True, alpha=0.2)
                                st.pyplot(fig_b2)
                            except Exception as ex:
                                st.error(f"Render Error: {ex}")

                        with col_p3:
                            st.write("📈 **Time-Offset Histogram**")
                            try:
                                if best_offsets_list and len(best_offsets_list) > 0:
                                    fig_b3, ax_b3 = plt.subplots(figsize=(4, 2.5))
                                    ax_b3.hist(best_offsets_list, bins=30, color='steelblue', alpha=0.7)
                                    ax_b3.set_xlabel("Offset Shift (s)", fontsize=6)
                                    ax_b3.set_ylabel("Hashes Count", fontsize=6)
                                    ax_b3.tick_params(labelsize=6)
                                    ax_b3.grid(True, alpha=0.2)
                                    st.pyplot(fig_b3)
                                else:
                                    st.caption("No consensus distribution match profile found.")
                            except Exception as ex:
                                st.error(f"Render Error: {ex}")

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

            df_results = pd.DataFrame(results)
            st.dataframe(df_results, use_container_width=True)

            csv_content = df_results.to_csv(index=False)

            st.download_button(
                label="📥 Download results.csv",
                data=csv_content,
                file_name="results.csv",
                mime="text/csv"
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Audited Files", len(results))
            with col2:
                unknowns = sum(1 for r in results if r['prediction'] in ['UNKNOWN', 'ERROR'])
                st.metric("Identified Tracks", len(results) - unknowns)
            with col3:
                errors = sum(1 for r in results if r['prediction'] == 'ERROR')
                st.metric("Execution Failures", errors)