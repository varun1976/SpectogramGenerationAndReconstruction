import streamlit as st
import numpy as np
import pandas as pd
from core.audio.reconstruction import reconstruct_audio
from core.metrics.snr import calculate_snr
from core.metrics.stoi import calculate_stoi
from core.metrics.pesq import calculate_pesq
from app.components.audio_player import render_audio_player, get_audio_download_bytes
from app.components.metrics_table import render_metrics_table, render_csv_download
from app.config import RECON_METHODS

def render_reconstruction_page():
    st.title("🧩 Audio Reconstruction & Phase Retrieval")
    st.markdown("""
    When converting an audio signal to a magnitude spectrogram, phase information is discarded. 
    Phase retrieval algorithms estimate lost spectral phase from magnitude spectrograms prior to inverse STFT (ISTFT).
    
    Compare reconstruction algorithms:
    - **Griffin-Lim Algorithm**: Iterative magnitude projection algorithm.
    - **SPSI (Single-Pass Single-Sideband)**: Peak-based phase accumulator algorithm.
    - **Hybrid SPSI + Griffin-Lim**: Uses SPSI as phase initializer followed by Griffin-Lim iterations.
    - **Random Phase Baseline**: Uniform random phase initialization.
    """)

    if "y" not in st.session_state or st.session_state["y"] is None:
        st.warning("⚠️ Please select/upload an audio file in the **2. Spectrogram Generation** page first.")
        return

    y = st.session_state["y"]
    sr = st.session_state["sr"]
    n_fft = st.session_state.get("n_fft", 1024)
    hop_length = st.session_state.get("hop_length", 256)
    win_length = st.session_state.get("win_length", n_fft)
    window_type = st.session_state.get("window_type", "hann")
    mag_sgram = st.session_state.get("mag_sgram", None)

    if mag_sgram is None:
        st.warning("⚠️ Spectrogram data is missing. Please visit page 2 first.")
        return

    st.sidebar.header("⚙️ Reconstruction Controls")
    selected_method_name = st.sidebar.selectbox("Reconstruction Method", options=list(RECON_METHODS.keys()), index=0)
    method_code = RECON_METHODS[selected_method_name]

    if method_code in ["griffin_lim", "spsi_gl"]:
        slider_label = "Griffin-Lim Iterations" if method_code == "griffin_lim" else "Hybrid (SPSI+GL) Iterations"
        n_iter = st.sidebar.slider(slider_label, 5, 100, 50, 5)
    else:
        st.sidebar.caption(f"ℹ️ Iterations do not apply to {selected_method_name} (1-pass algorithm).")
        n_iter = 1

    st.subheader(f"1. Active Method: {selected_method_name}")
    
    if st.button("▶️ Synthesize Audio", type="primary"):
        with st.spinner("Synthesizing audio signal from magnitude spectrogram..."):
            y_recon = reconstruct_audio(
                msgram=mag_sgram,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window_type,
                method=method_code,
                n_iter=n_iter,
                target_length=len(y)
            )
            st.session_state["single_recon_y"] = y_recon

    if "single_recon_y" in st.session_state:
        y_recon = st.session_state["single_recon_y"]
        try:
            snr_val = calculate_snr(y, y_recon)
        except Exception:
            snr_val = 0.0
            
        try:
            stoi_val = calculate_stoi(y, y_recon, sr)
        except Exception:
            stoi_val = 0.0
            
        try:
            pesq_val = calculate_pesq(y, y_recon, sr)
        except Exception:
            pesq_val = 0.0

        col1, col2, col3 = st.columns(3)
        col1.metric("SNR (Signal-to-Noise Ratio)", f"{round(snr_val, 2)} dB")
        col2.metric("PESQ Score", f"{round(pesq_val, 2)}")
        col3.metric("STOI Intelligibility", f"{round(stoi_val, 4)}")

        render_audio_player(y_recon, sr, label=f"Synthesized Audio ({selected_method_name})")

        wav_bytes = get_audio_download_bytes(y_recon, sr)
        st.download_button(
            label="📥 Download Reconstructed WAV File",
            data=wav_bytes,
            file_name=f"reconstructed_{method_code}.wav",
            mime="audio/wav"
        )

    st.markdown("---")
    st.subheader("2. Compare All Phase Retrieval Methods")
    if st.button("🔄 Benchmark All Phase Retrieval Algorithms"):
        with st.spinner("Benchmarking Griffin-Lim, SPSI, Hybrid, and Random Phase..."):
            bench_results = []
            audio_map = {}
            for name, code in RECON_METHODS.items():
                y_r = reconstruct_audio(
                    msgram=mag_sgram,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    window=window_type,
                    method=code,
                    n_iter=n_iter,
                    target_length=len(y)
                )
                try:
                    snr_v = calculate_snr(y, y_r)
                except Exception:
                    snr_v = 0.0
                    
                try:
                    stoi_v = calculate_stoi(y, y_r, sr)
                except Exception:
                    stoi_v = 0.0
                    
                try:
                    pesq_v = calculate_pesq(y, y_r, sr)
                except Exception:
                    pesq_v = 0.0
                
                bench_results.append({
                    "Algorithm": name,
                    "SNR (dB)": round(snr_v, 2),
                    "PESQ": round(pesq_v, 2),
                    "STOI": round(stoi_v, 4)
                })
                audio_map[name] = y_r
                
            df_bench = pd.DataFrame(bench_results)
            st.session_state["df_bench"] = df_bench
            st.session_state["audio_map"] = audio_map

    if "df_bench" in st.session_state:
        df_bench = st.session_state["df_bench"]
        audio_map = st.session_state["audio_map"]

        render_metrics_table(df_bench, title="Phase Retrieval Benchmark Comparison")
        render_csv_download(df_bench, filename="phase_retrieval_benchmark.csv")

        st.markdown("#### Audio Playback Comparison")
        for name, y_r in audio_map.items():
            render_audio_player(y_r, sr, label=name)
