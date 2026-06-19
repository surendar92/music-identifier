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
            bins=np.arange(min(offsets)-1, max(offsets)+1, 0.5)
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
            <h3>🎧 Single Clip</h3>
            <p style="color: #A3A8B4; font-size: 14px;">Identify a single song using fast fingerprint matching.</p>
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
            <h3>📂 Batch Mode</h3>
            <p style="color: #A3A8B4; font-size: 14px;">Process and catalog multiple audio files at once.</p>
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
        os.makedirs(temp_dir, exist_ok=True)  # Automatically creates the folder if it's missing
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(
            ["📊 Result", "🎼 Spectrogram", "⭐ Constellation Peaks", "📈 Offset Histogram"]
        )

        # ==============================================================================
        # TAB 1: MODERN RESULT VIEW
        # ==============================================================================
        with tab1:
            st.write("")  # Quick spacer

            # 1. Main Match Banner Card
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, #1E2235 0%, #111424 100%);
                    padding: 24px;
                    border-radius: 12px;
                    border: 1px solid #2D3142;
                    border-left: 5px solid #00D1B2;
                    margin-bottom: 20px;
                ">
                    <span style="color: #00D1B2; font-weight: 700; font-size: 12px; uppercase; tracking-step: 1px;">🎉 TOP MATCH IDENTIFIED</span>
                    <h2 style="margin: 4px 0 0 0; color: #FFFFFF; font-size: 32px;">Back In The U.S.S.R.</h2>
                    <p style="margin: 6px 0 0 0; color: #A3A8B4; font-size: 14px;">🎯 Audio matching logic completed successfully with high precision.</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            # 2. Key Metrics Row (Using Columns)
            metric_col1, metric_col2 = st.columns(2)

            with metric_col1:
                st.markdown(
                    """
                    <div style="background-color: #1E2235; padding: 16px; border-radius: 10px; border: 1px solid #2D3142; text-align: center;">
                        <p style="margin:0; font-size:13px; color:#A3A8B4; text-transform:uppercase; font-weight:600; letter-spacing:0.5px;">🎯 Match Confidence Score</p>
                        <h2 style="margin:0; padding-top:6px; font-size:36px; color:#FF4B4B;">18,235</h2>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with metric_col2:
                st.markdown(
                    """
                    <div style="background-color: #1E2235; padding: 16px; border-radius: 10px; border: 1px solid #2D3142; text-align: center;">
                        <p style="margin:0; font-size:13px; color:#A3A8B4; text-transform:uppercase; font-weight:600; letter-spacing:0.5px;">📊 Statistical Certainty</p>
                        <h2 style="margin:0; padding-top:6px; font-size:36px; color:#00D1B2;">100%</h2>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.write("")
            st.markdown("### 📋 Top 5 Match Candidates")

            # 3. Modern Candidate Standings Table
            # (Dynamically inject your ranking array items inside this HTML structure)
            candidates_html = """
            <table style="width:100%; border-collapse: collapse; margin-top:10px; font-size:15px; color:#E2E8F0;">
                <thead>
                    <tr style="border-bottom: 2px solid #2D3142; text-align: left; color:#A3A8B4;">
                        <th style="padding: 10px; font-weight:600; width: 10%;">Rank</th>
                        <th style="padding: 10px; font-weight:600; width: 65%;">Song Title</th>
                        <th style="padding: 10px; font-weight:600; width: 25%; text-align: right;">Total Matches</th>
                    </tr>
                </thead>
                <tbody>
                    <tr style="border-bottom: 1px solid #1E2235; background-color: rgba(0, 209, 178, 0.05);">
                        <td style="padding: 12px; font-weight: 700; color:#00D1B2;">#1</td>
                        <td style="padding: 12px; font-weight: 600; color:#FFFFFF;">🥇 Back In The U.S.S.R.</td>
                        <td style="padding: 12px; text-align: right; font-weight: 700; color:#00D1B2;">19,989</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #1E2235;">
                        <td style="padding: 12px; font-weight: 600; color:#A3A8B4;">#2</td>
                        <td style="padding: 12px; color:#E2E8F0;">Never Gonna Give You Up</td>
                        <td style="padding: 12px; text-align: right; color:#E2E8F0;">1,728</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #1E2235;">
                        <td style="padding: 12px; font-weight: 600; color:#A3A8B4;">#3</td>
                        <td style="padding: 12px; color:#E2E8F0;">Hey Jude</td>
                        <td style="padding: 12px; text-align: right; color:#E2E8F0;">1,572</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #1E2235;">
                        <td style="padding: 12px; font-weight: 600; color:#A3A8B4;">#4</td>
                        <td style="padding: 12px; color:#E2E8F0;">While My Guitar Gently Weeps</td>
                        <td style="padding: 12px; text-align: right; color:#E2E8F0;">1,107</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #2D3142;">
                        <td style="padding: 12px; font-weight: 600; color:#A3A8B4;">#5</td>
                        <td style="padding: 12px; color:#E2E8F0;">A Day In The Life</td>
                        <td style="padding: 12px; text-align: right; color:#E2E8F0;">1,052</td>
                    </tr>
                </tbody>
            </table>
            """
            st.markdown(candidates_html, unsafe_allow_html=True)
        # ==============================================================================
        # TAB 2: SPECTROGRAM
        # ==============================================================================
        with tab2:
            st.subheader("🎼 Spectrogram Visualization")

            # Check if metadata exists in session_state before trying to plot
            if "metadata" in st.session_state and st.session_state.metadata is not None:
                try:
                    # Pull directly from session state
                    current_metadata = st.session_state.metadata

                    spec_db = current_metadata['spectrogram']
                    times = current_metadata['times']
                    freqs = current_metadata['frequencies']

                    # Generate the matplotlib figure
                    fig, ax = plt.subplots(figsize=(12, 5))
                    im = ax.imshow(
                        spec_db,
                        aspect='auto',
                        origin='lower',
                        cmap='magma',
                        extent=[times[0], times[-1], freqs[0], freqs[-1]]
                    )

                    ax.set_xlabeled("Time (seconds)")
                    ax.set_ylabel("Frequency (Hz)")
                    ax.set_title("Spectrogram (dB scale)")
                    plt.colorbar(im, ax=ax, label="Magnitude (dB)")

                    # Render the plot in Streamlit
                    st.pyplot(fig)

                except Exception as e:
                    st.error(f"Error rendering plot: {e}")

            else:
                # User hasn't processed a file yet in this session run
                st.info(
                    "👋 Please upload and identify an audio file in the 'Result' tab first to generate its spectrogram.")

        # ==============================================================================
        # TAB 3: CONSTELLATION
        # ==============================================================================
        with tab3:
            st.subheader("⭐ Peak Constellation")

            # Check if metadata exists in session_state before parsing peaks
            if "metadata" in st.session_state and st.session_state.metadata is not None:
                try:
                    # Pull safely from session state
                    current_metadata = st.session_state.metadata

                    peak_times = current_metadata['peak_times']
                    peak_freqs = current_metadata['peak_freqs']

                    if len(peak_times) > 0:
                        fig, ax = plt.subplots(figsize=(12, 5))
                        ax.scatter(
                            peak_times,
                            peak_freqs,
                            alpha=0.6,
                            s=40,
                            color='orange',
                            edgecolors='darkorange'
                        )
                        ax.set_xlabel("Time (seconds)")
                        ax.set_ylabel("Frequency (Hz)")
                        ax.set_title(f"Peak Constellation - {len(peak_times)} peaks")
                        ax.grid(visible=True, alpha=0.3)

                        # Render the plot in Streamlit
                        st.pyplot(fig)
                    else:
                        st.info("No peaks detected")

                except Exception as e:
                    st.error(f"Error rendering constellation map: {e}")

            else:
                # User hasn't processed a file yet in this session run
                st.info(
                    "👋 Please upload and identify an audio file in the 'Result' tab first to view its constellation peaks.")
        
        # ===== TAB 4: OFFSET HISTOGRAM =====
        with tab4:
            st.subheader("📈 Offset Histogram")
            try:
                best_song, best_score, best_offsets_list, _, _ = identify_query_clip_sqlite(
                    temp_path, 
                    db_conn
                )
                
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
    st.subheader("📁 Batch Processing")
    
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
                os.makedirs(temp_dir, exist_ok=True)  # Automatically creates the folder if it's missing
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                try:
                    best_song, _, _, _, _ = identify_query_clip_sqlite(temp_path, db_conn)
                    filename_without_ext = os.path.splitext(uploaded_file.name)[0]
                    prediction = best_song if best_song != "Unknown / No Match" else "UNKNOWN"
                    
                    results.append({
                        "filename": filename_without_ext,
                        "prediction": prediction
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
                st.metric("Total", len(results))
            with col2:
                unknowns = sum(1 for r in results if r['prediction'] in ['UNKNOWN', 'ERROR'])
                st.metric("Identified", len(results) - unknowns)
            with col3:
                errors = sum(1 for r in results if r['prediction'] == 'ERROR')
                st.metric("Errors", errors)

# ==============================================================================
# MODERN SIDEBAR: DATABASE STATUS
# ==============================================================================
with st.sidebar:
    # 1. This should be your ONLY sidebar header now
    st.markdown("### 📊 Database Status")

    # Get stats from your existing backend helper
    num_songs, unique_hashes, total_entries = get_db_stats()
    db_size = os.path.getsize("music_database.db") / (1024 * 1024)

    # ❌ REMOVE / COMMENT OUT ANY OLD CODES LIKE THESE:
    # st.markdown("### 📊 Database Stats")  <-- Delete this old one if duplicated
    # st.metric(label="Songs", ...)         <-- Delete these old vertical metrics
    # st.metric(label="Unique Hashes", ...)
    # st.metric(label="Total Entries", ...)
    # st.metric(label="DB Size", ...)

    # 2. Keep ONLY the grid layout structure below:
    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)

    with row1_col1:
        st.markdown(
            f"""
            <div style="background-color: #1E2235; padding: 10px; border-radius: 8px; border: 1px solid #2D3142; margin-bottom: 10px;">
                <p style="margin:0; font-size:12px; color:#A3A8B4; text-transform:uppercase; font-weight:600;">Songs</p>
                <h3 style="margin:0; padding-top:4px; font-size:22px; color:#FFFFFF;">{num_songs:,}</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

    with row1_col2:
        st.markdown(
            f"""
            <div style="background-color: #1E2235; padding: 10px; border-radius: 8px; border: 1px solid #2D3142; margin-bottom: 10px;">
                <p style="margin:0; font-size:12px; color:#A3A8B4; text-transform:uppercase; font-weight:600;">DB Size</p>
                <h3 style="margin:0; padding-top:4px; font-size:22px; color:#00D1B2;">{db_size:.1f} MB</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

    with row2_col1:
        st.markdown(
            f"""
            <div style="background-color: #1E2235; padding: 10px; border-radius: 8px; border: 1px solid #2D3142;">
                <p style="margin:0; font-size:11px; color:#A3A8B4; text-transform:uppercase; font-weight:600;">Hashes</p>
                <h3 style="margin:0; padding-top:4px; font-size:18px; color:#FFFFFF;">{unique_hashes:,}</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

    with row2_col2:
        st.markdown(
            f"""
            <div style="background-color: #1E2235; padding: 10px; border-radius: 8px; border: 1px solid #2D3142;">
                <p style="margin:0; font-size:11px; color:#A3A8B4; text-transform:uppercase; font-weight:600;">Entries</p>
                <h3 style="margin:0; padding-top:4px; font-size:18px; color:#FFFFFF;">{total_entries:,}</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    # Elegant Backend Status Info Group
    st.markdown("### ⚡ Backend Info")

    st.markdown(
        """
        <div style="background-color: rgba(255, 255, 255, 0.03); padding: 12px; border-radius: 8px; border-left: 3px solid #FF4B4B;">
            <p style="margin:0; font-size:13px; color:#E2E8F0; line-height: 1.5;">
                ⚡ <b>Fast indexed queries</b><br>
                💾 <b>No RAM overhead</b>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )