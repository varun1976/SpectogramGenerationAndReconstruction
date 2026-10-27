import sys
import os

# Add root directory to sys.path to enable core and experiments imports
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
from app.components.theme import apply_custom_theme
from app.pages.dashboard import render_dashboard_page
from app.pages.spectrogram import render_spectrogram_page
from app.pages.compression import render_compression_page
from app.pages.cqt_experiment import render_cqt_page
from app.pages.png_to_wav import render_png_to_wav_page
from app.pages.reconstruction import render_reconstruction_page

# Set Streamlit page configuration
st.set_page_config(
    page_title="Spectrogram Generation & Reconstruction Platform",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    # Inject clean glassmorphic sidebar CSS
    apply_custom_theme()

    # Sidebar Header / Branding
    st.sidebar.markdown(
        """
        <div class="sidebar-brand-container">
            <h2 class="sidebar-brand-title">🎵 SpectroLab</h2>
            <div class="sidebar-brand-subtitle">SPECTROGRAM RESEARCH PLATFORM</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    page_choice = st.sidebar.radio(
        "Navigation",
        [
            "1. Dashboard",
            "2. Spectrogram Generation",
            "3. Image Compression Experiment",
            "4. CQT Experiment",
            "5. PNG → WAV Reconstruction",
            "6. Audio Reconstruction"
        ],
        label_visibility="collapsed"
    )

    # Sidebar Footer Project Info Card
    st.sidebar.markdown(
        """
        <div class="sidebar-footer-card">
            <div class="sidebar-footer-title">Research Project</div>
            <div class="sidebar-footer-desc">
                Evaluating Image Quantization & CQT Mapping Losses
            </div>
            <div class="sidebar-status-badge">
                <span class="status-dot"></span> System Ready • v2.0
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if page_choice.startswith("1."):
        render_dashboard_page()
    elif page_choice.startswith("2."):
        render_spectrogram_page()
    elif page_choice.startswith("3."):
        render_compression_page()
    elif page_choice.startswith("4."):
        render_cqt_page()
    elif page_choice.startswith("5."):
        render_png_to_wav_page()
    elif page_choice.startswith("6."):
        render_reconstruction_page()

if __name__ == "__main__":
    main()
