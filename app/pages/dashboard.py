import streamlit as st
from app.components.pipeline_view import render_pipeline_diagram

def render_dashboard_page():
    st.title("🔬 Spectrogram Generation & Reconstruction Research Platform")
    st.markdown(
        "**Research Title:** *Spectrogram Generation and Reconstruction: Evaluating Image Quantization and CQT Mapping Losses*"
    )
    
    st.markdown("---")
    
    # 1. Project Motivation & Objectives (Full Width)
    st.subheader("📌 Project Motivation & Objectives")
    st.markdown("""
    Spectrograms are widely used as visual 2D image representations of 1D temporal audio signals in machine learning, audio synthesis, and classification models.
    However, converting continuous Fourier spectra into 8-bit image formats (PNG/JPEG) and log-frequency (Constant-Q) representations introduces irreversible acoustic information loss.
    
    This web application enables empirical evaluation across three core research questions:
    1. **Research Question 1 (Quantization Loss)**: How much acoustic degradation is introduced by mapping continuous spectral magnitudes into 8-bit `[0, 255]` integer values?
    2. **Research Question 2 (Compression Loss)**: How does JPEG lossy compression (quality levels 100%, 80%, 50%) degrade reconstructed audio quality compared to lossless PNG?
    3. **Research Question 3 (CQT Mapping Loss)**: How much information is lost during non-linear `Linear -> Constant-Q (CQT) -> Linear` transformations?
    """)
    
    st.markdown("---")
    
    # 2. Quick-Start Guide (Full Width)
    st.subheader("🚀 Quick-Start Guide")
    st.markdown("""
    1. Navigate to **2. Spectrogram Generation** to upload a `.wav` file or generate sample audio.
    2. Adjust STFT window parameters (`n_fft`, `hop_length`, `win_length`, `window`) and observe the generated 8-bit spectrogram.
    3. Run **3. Image Compression Experiment** to compare PNG against JPEG 100, 80, and 50 compression quality.
    4. Run **4. CQT Roundtrip Experiment** to analyze Linear -> CQT -> Linear spectral and acoustic loss.
    5. Use **5. PNG → WAV Reconstruction** to invert raw PNG spectrogram images back into playable audio.
    6. Explore **6. Audio Reconstruction** to benchmark phase retrieval algorithms (Griffin-Lim, SPSI, Hybrid SPSI+GL).
    """)

    st.markdown("---")

    # 3. System Pipeline Overview (Bottom Full Width)
    st.subheader("⚙️ System Pipeline Overview")
    render_pipeline_diagram()
