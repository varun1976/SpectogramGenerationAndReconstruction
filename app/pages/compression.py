import os
import streamlit as st
import numpy as np
import pandas as pd
from experiments.compression import run_compression_experiment
from app.components.plots import plot_spectrogram, plot_spectrogram_comparison, plot_difference_heatmap
from app.components.audio_player import render_audio_player, get_audio_download_bytes
from app.components.metrics_table import render_metrics_table, render_csv_download

def render_compression_page():
    st.title("📉 Image Compression Experiment (PNG vs JPEG)")
    st.markdown("""
    **Research Question 2:** *How does JPEG lossy compression affect the quality of reconstructed audio compared with lossless PNG?*
    
    This experiment evaluates 8-bit spectrogram images saved under 4 formats:
    - **PNG**: Lossless 8-bit reference
    - **JPEG 100**: Maximum JPEG quality
    - **JPEG 80**: Standard web JPEG compression
    - **JPEG 50**: Heavy JPEG compression
    """)

    if "y" not in st.session_state or st.session_state["y"] is None:
        st.warning("⚠️ Please select/upload an audio file in the **2. Spectrogram Generation** page first.")
        return

    y = st.session_state["y"]
    sr = st.session_state["sr"]
    n_fft = st.session_state.get("n_fft", 1024)
    hop_length = st.session_state.get("hop_length", 256)
    scale_type = st.session_state.get("scale_type", "log1p")

    st.sidebar.header("⚙️ Experiment Controls")
    recon_method = st.sidebar.selectbox("Reconstruction Algorithm", options=["griffin_lim", "spsi", "spsi_gl"], index=0)

    if recon_method in ["griffin_lim", "spsi_gl"]:
        slider_label = "Griffin-Lim Iterations" if recon_method == "griffin_lim" else "Hybrid (SPSI+GL) Iterations"
        n_iter = st.sidebar.slider(slider_label, 10, 100, 50, 10)
    else:
        st.sidebar.caption(f"ℹ️ Iterations do not apply to SPSI (1-pass algorithm).")
        n_iter = 1

    if st.button("▶️ Run Image Compression Experiment", type="primary"):
        with st.spinner("Processing PNG and JPEG compression pipelines..."):
            df_results, exp_data = run_compression_experiment(
                y=y,
                sr=sr,
                n_fft=n_fft,
                hop_length=hop_length,
                scale_type=scale_type,
                recon_method=recon_method,
                n_iter=n_iter
            )
            st.session_state["comp_df"] = df_results
            st.session_state["comp_exp_data"] = exp_data
            st.success("Experiment completed!")

    if "comp_df" in st.session_state:
        df_results = st.session_state["comp_df"]
        exp_data = st.session_state["comp_exp_data"]

        render_metrics_table(df_results, title="Image Compression Metric Comparison")
        render_csv_download(df_results, filename="compression_experiment_results.csv")

        st.markdown("---")
        st.subheader("🔍 Visual & Acoustic Format Inspector")
        
        selected_fmt = st.selectbox("Select Format to Inspect:", options=list(exp_data["artifacts"].keys()), index=0)
        art = exp_data["artifacts"][selected_fmt]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Loaded {selected_fmt} Spectrogram Image**")
            fig_img = plot_spectrogram(art["img_array"], sr, hop_length, title=f"{selected_fmt} Image [0-255]", is_uint8=True)
            st.pyplot(fig_img)

        with col2:
            st.markdown(f"**Spectrogram Difference Heatmap (|Original - {selected_fmt}|)**")
            diff_mag = np.abs(exp_data["original_mag"] - art["dequant_mag"])
            fig_diff = plot_difference_heatmap(diff_mag, sr, hop_length, title=f"Absolute Error: {selected_fmt}")
            st.pyplot(fig_diff)

        st.markdown("---")
        render_audio_player(art["reconstructed_audio"], sr, label=f"Reconstructed Audio ({selected_fmt})")
        
        wav_bytes = get_audio_download_bytes(art["reconstructed_audio"], sr)
        st.download_button(
            label=f"📥 Download Reconstructed WAV ({selected_fmt})",
            data=wav_bytes,
            file_name=f"reconstructed_{selected_fmt.lower().replace(' ', '_')}.wav",
            mime="audio/wav"
        )
