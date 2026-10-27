import os
import json
import io
import streamlit as st
import numpy as np
from PIL import Image, PngImagePlugin
from core.audio.loader import load_audio
from core.spectrogram.generator import generate_spectrogram
from core.image.preprocessing import array_to_image
from app.components.plots import plot_waveform, plot_spectrogram
from app.components.audio_player import render_audio_player, get_audio_download_bytes
from app.config import INPUT_DIR, DEFAULT_CONFIG, WINDOW_FUNCTIONS

def generate_synthetic_audio(duration: float = 2.0, sr: int = 22050) -> np.ndarray:
    """Generate a clean synthetic harmonic test signal (440Hz A4 + 880Hz A5 chirp)."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Sine wave mixture with frequency sweep
    f_t = np.linspace(220, 880, len(t))
    y = 0.5 * np.sin(2 * np.pi * f_t * t) + 0.3 * np.sin(2 * np.pi * 440 * t)
    return y.astype(np.float32)

def render_spectrogram_page():
    st.title("🎛️ Audio Input & Spectrogram Generation")
    
    st.sidebar.header("⚙️ STFT Parameters")
    n_fft = st.sidebar.select_slider("FFT Size (n_fft)", options=[256, 512, 1024, 2048, 4096], value=1024)
    hop_length = st.sidebar.select_slider("Hop Length", options=[64, 128, 256, 512, 1024], value=256)
    win_length = st.sidebar.select_slider("Window Length", options=[256, 512, 1024, 2048], value=n_fft if n_fft <= 2048 else 2048)
    window_type = st.sidebar.selectbox("Window Function", options=WINDOW_FUNCTIONS, index=0)
    scale_type = st.sidebar.selectbox("Magnitude Scaling", options=["log1p", "db"], index=0)
    
    st.subheader("1. Audio Source Selection")
    source_choice = st.radio("Choose input source:", ["Upload WAV File", "Generate Synthetic Test Signal", "Use Sample Audio"], horizontal=True)
    
    y, sr, meta = None, 22050, {}
    
    if source_choice == "Upload WAV File":
        uploaded_file = st.file_uploader("Upload .wav file", type=["wav"])
        if uploaded_file is not None:
            temp_path = os.path.join(INPUT_DIR, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            try:
                y, sr, meta = load_audio(temp_path)
                st.success(f"Loaded '{uploaded_file.name}' successfully.")
            except Exception as e:
                st.error(f"Error loading uploaded file: {str(e)}")
        else:
            st.info("Please upload a .wav audio file to proceed.")
            return
    elif source_choice == "Generate Synthetic Test Signal":
        duration = st.slider("Signal Duration (seconds)", 1.0, 5.0, 2.0, 0.5)
        y = generate_synthetic_audio(duration=duration, sr=22050)
        sr = 22050
        meta = {
            "filename": "synthetic_harmonic_test.wav",
            "sample_rate": sr,
            "duration": duration,
            "n_samples": len(y),
            "n_channels": 1
        }
    else:
        # Use built-in generated audio
        y = generate_synthetic_audio(duration=3.0, sr=22050)
        sr = 22050
        meta = {
            "filename": "sample_audio.wav",
            "sample_rate": sr,
            "duration": 3.0,
            "n_samples": len(y),
            "n_channels": 1
        }

    if y is None or len(y) == 0:
        return

    # Store loaded audio in Streamlit session state
    st.session_state["y"] = y
    st.session_state["sr"] = sr
    st.session_state["n_fft"] = n_fft
    st.session_state["hop_length"] = hop_length
    st.session_state["win_length"] = win_length
    st.session_state["window_type"] = window_type
    st.session_state["scale_type"] = scale_type
    
    # Display audio properties
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Sample Rate", f"{sr} Hz")
    col_b.metric("Duration", f"{round(len(y)/sr, 2)} s")
    col_c.metric("Samples", f"{len(y)}")
    col_d.metric("Channels", f"{meta.get('n_channels', 1)}")

    render_audio_player(y, sr, label="Original Audio Playback")

    # Plot Waveform
    fig_wave = plot_waveform(y, sr, title=f"Audio Waveform - {meta.get('filename', 'Audio')}")
    st.pyplot(fig_wave)

    st.markdown("---")
    st.subheader("2. Spectrogram Generation & 8-bit Quantization")
    
    stft_complex, mag_sgram, scaled_sgram, uint8_img, q_meta = generate_spectrogram(
        y=y,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window_type,
        scale_type=scale_type
    )

    st.session_state["mag_sgram"] = mag_sgram
    st.session_state["scaled_sgram"] = scaled_sgram
    st.session_state["uint8_img"] = uint8_img
    # Build raw PNG file bytes (clean image matrix without Matplotlib plot axes/colorbar)
    img_pil = array_to_image(uint8_img, flip_vertical=True)
    buf_png = io.BytesIO()
    info = PngImagePlugin.PngInfo()
    info.add_text("meta", json.dumps(q_meta))
    img_pil.save(buf_png, format="PNG", pnginfo=info)
    png_bytes = buf_png.getvalue()
    json_bytes = json.dumps(q_meta, indent=2).encode("utf-8")
    
    st.markdown("### 📥 Download Clean Spectrogram Artifacts")
    st.info("Download the raw 8-bit PNG image (containing the exact pixel matrix without plot axes) and JSON metadata to use in the **PNG → WAV Reconstruction** page.")
    
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.download_button(
            label="📥 Download Raw 8-bit Spectrogram (PNG)",
            data=png_bytes,
            file_name="spectrogram.png",
            mime="image/png",
            type="primary"
        )
    with col_d2:
        st.download_button(
            label="📥 Download Metadata (JSON)",
            data=json_bytes,
            file_name="spectrogram_metadata.json",
            mime="application/json"
        )
        
    st.markdown("---")
    st.subheader("3. Spectrogram Visualizations")

    tab1, tab2, tab3 = st.tabs(["Linear Magnitude Spectrogram", "Scaled Spectrogram", "8-bit Quantized PNG Image"])
    
    with tab1:
        fig_mag = plot_spectrogram(mag_sgram, sr, hop_length, title="Linear Magnitude Spectrogram", cmap="viridis")
        st.pyplot(fig_mag)
        
    with tab2:
        fig_scale = plot_spectrogram(scaled_sgram, sr, hop_length, title=f"Scaled Spectrogram ({scale_type})", cmap="magma")
        st.pyplot(fig_scale)
        
    with tab3:
        fig_quant = plot_spectrogram(uint8_img, sr, hop_length, title="8-bit Quantized Spectrogram Image [0-255]", is_uint8=True)
        st.pyplot(fig_quant)
        st.json(q_meta)
