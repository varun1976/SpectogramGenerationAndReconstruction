import streamlit as st
import numpy as np
import pandas as pd
from experiments.cqt_roundtrip import run_cqt_roundtrip_experiment
from app.components.plots import plot_spectrogram, plot_difference_heatmap
from app.components.audio_player import render_audio_player, get_audio_download_bytes
from app.components.metrics_table import render_metrics_table, render_csv_download

def render_cqt_page():
    st.title("🎼 Constant-Q Transform (CQT) Mapping Experiment")
    st.markdown("""
    **Research Question 3:** *How much additional information is lost when a spectrogram undergoes Linear -> CQT -> Linear roundtrip mapping?*
    
    This pipeline maps linear-frequency STFT bins into a logarithmic frequency (CQT) space using Dan Ellis' `logfmap` transformation matrix, quantizes to an 8-bit image, and reverses the transformation back to linear frequency space prior to phase retrieval and audio reconstruction.
    """)

    if "y" not in st.session_state or st.session_state["y"] is None:
        st.warning("⚠️ Please select/upload an audio file in the **2. Spectrogram Generation** page first.")
        return

    y = st.session_state["y"]
    sr = st.session_state["sr"]
    n_fft = st.session_state.get("n_fft", 1024)
    hop_length = st.session_state.get("hop_length", 256)
    scale_type = st.session_state.get("scale_type", "log1p")

    st.sidebar.header("⚙️ CQT Parameters")
    low_row = st.sidebar.slider("Lowest FFT Row (lowRow)", 1, 50, 10, 1)
    cqt_bins = st.sidebar.select_slider("CQT Frequency Bins", options=[128, 256, 513, 1024], value=513)
    recon_method = st.sidebar.selectbox("Reconstruction Algorithm", options=["griffin_lim", "spsi", "spsi_gl"], index=0)

    if recon_method in ["griffin_lim", "spsi_gl"]:
        slider_label = "Griffin-Lim Iterations" if recon_method == "griffin_lim" else "Hybrid (SPSI+GL) Iterations"
        n_iter = st.sidebar.slider(slider_label, 10, 100, 50, 10)
    else:
        st.sidebar.caption(f"ℹ️ Iterations do not apply to SPSI (1-pass algorithm).")
        n_iter = 1

    if st.button("▶️ Run CQT Roundtrip Experiment", type="primary"):
        with st.spinner("Processing Linear -> CQT -> Linear roundtrip pipeline..."):
            results = run_cqt_roundtrip_experiment(
                y=y,
                sr=sr,
                n_fft=n_fft,
                hop_length=hop_length,
                scale_type=scale_type,
                low_row=low_row,
                cqt_bins=cqt_bins,
                recon_method=recon_method,
                n_iter=n_iter
            )
            st.session_state["cqt_results"] = results
            st.success("CQT roundtrip experiment complete!")

    if "cqt_results" in st.session_state:
        res = st.session_state["cqt_results"]
        metrics = res["metrics"]

        # Results Summary Metrics
        st.subheader("📊 CQT Metric Summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Direct Linear SNR", f"{round(metrics['snr_direct_db'], 2)} dB")
        col2.metric("CQT Roundtrip SNR", f"{round(metrics['snr_cqt_db'], 2)} dB")
        col3.metric("CQT vs Direct SNR", f"{round(metrics['snr_cqt_vs_direct_db'], 2)} dB")
        col4.metric("Mean Spectral Error", f"{round(metrics['mean_diff'], 4)}")

        df_cqt = pd.DataFrame([{
            "Pipeline": "Direct Linear Baseline",
            "SNR (dB)": round(metrics["snr_direct_db"], 2),
            "PESQ": round(metrics["pesq_direct"], 2),
            "STOI": round(metrics["stoi_direct"], 4)
        }, {
            "Pipeline": "Linear -> CQT -> Linear",
            "SNR (dB)": round(metrics["snr_cqt_db"], 2),
            "PESQ": round(metrics["pesq_cqt"], 2),
            "STOI": round(metrics["stoi_cqt"], 4)
        }])
        render_metrics_table(df_cqt, title="Direct Linear vs CQT Roundtrip Evaluation")
        render_csv_download(df_cqt, filename="cqt_experiment_results.csv")

        st.markdown("---")
        st.subheader("🖼️ Spectrogram Visual Comparisons")
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "Original Linear Spectrogram",
            "CQT Spectrogram",
            "Reconstructed Linear Spectrogram",
            "Difference Heatmap"
        ])

        with tab1:
            fig1 = plot_spectrogram(res["original_mag_linear"], sr, hop_length, title="Original Linear Spectrogram", cmap="viridis")
            st.pyplot(fig1)

        with tab2:
            fig2 = plot_spectrogram(res["cqt_scaled"], sr, hop_length, title="CQT Spectrogram (Log Frequency)", cmap="inferno")
            st.pyplot(fig2)

        with tab3:
            fig3 = plot_spectrogram(res["recon_mag_linear"], sr, hop_length, title="Reconstructed Linear Spectrogram (from CQT)", cmap="viridis")
            st.pyplot(fig3)

        with tab4:
            fig4 = plot_difference_heatmap(res["diff_sgram"], sr, hop_length, title="Absolute Error: |Original - CQT Roundtrip|")
            st.pyplot(fig4)

        st.markdown("---")
        st.subheader("🔊 Audio Playback & Comparison")
        col_audio1, col_audio2 = st.columns(2)
        with col_audio1:
            render_audio_player(res["y_direct_recon"], sr, label="Direct Linear Reconstruction")
        with col_audio2:
            render_audio_player(res["y_cqt_recon"], sr, label="CQT Roundtrip Reconstruction")
