import os
import io
import json
import streamlit as st
import numpy as np
import pandas as pd
from core.image.png import load_and_validate_png
from core.audio.png2wav_engine import reconstruct_audio_from_png
from core.audio.loader import load_audio
from core.metrics.snr import calculate_snr
from core.metrics.stoi import calculate_stoi
from core.metrics.pesq import calculate_pesq
from app.components.plots import plot_spectrogram, plot_waveform
from app.components.audio_player import render_audio_player, get_audio_download_bytes
from app.config import WINDOW_FUNCTIONS, RECON_METHODS

def render_png_to_wav_page():
    st.title("🖼️ PNG → WAV Spectrogram Reconstruction")
    st.markdown("""
    **Spectrogram Inversion & Audio Recovery**
    
    Upload an 8-bit PNG spectrogram image and reconstruct a playable, downloadable WAV audio file.
    
    - **Mode A (PNG + Metadata)**: Automatically extracts original scaling and STFT parameters from embedded PNG text chunks (`PngInfo`) or an uploaded `.json` metadata file.
    - **Mode B (PNG Only)**: Manually configure STFT parameters (Sample Rate, Hop Length, Scale Min, Scale Max) when metadata is unavailable.
    """)

    st.subheader("1. File Uploads")
    col_up1, col_up2 = st.columns(2)
    
    with col_up1:
        uploaded_png = st.file_uploader("Upload PNG Spectrogram Image (.png)", type=["png"])
        
    with col_up2:
        uploaded_json = st.file_uploader("Upload Optional JSON Metadata File (.json)", type=["json"])

    if uploaded_png is None:
        st.info("👆 Please upload a PNG spectrogram image to proceed.")
        return

    # Validate image file and extract metadata
    try:
        bytes_data = uploaded_png.getvalue()
        png_stream = io.BytesIO(bytes_data)
        uint8_arr, embedded_meta, file_size, image_info = load_and_validate_png(png_stream)
    except Exception as e:
        st.error(f"❌ Image Validation Error: {str(e)}")
        return

    # Display Image Validation Info
    col_i1, col_i2, col_i3, col_i4 = st.columns(4)
    col_i1.metric("Dimensions", f"{image_info['width']} x {image_info['height']}")
    col_i2.metric("Color Mode", f"{image_info['original_mode']}")
    col_i3.metric("File Size", f"{round(file_size / 1024.0, 1)} KB")
    col_i4.metric("Embedded Metadata", "Found ✅" if image_info['has_embedded_metadata'] else "None ❌")

    if image_info['original_mode'] not in ['L', '1']:
        st.warning(f"⚠️ Input image format is '{image_info['original_mode']}'. It was automatically converted to 8-bit grayscale for spectrogram processing.")

    # Parse uploaded JSON metadata if provided
    external_meta = {}
    if uploaded_json is not None:
        try:
            external_meta = json.loads(uploaded_json.getvalue().decode("utf-8"))
            st.success("Loaded sidecar JSON metadata successfully.")
        except Exception as e:
            st.error(f"Error parsing uploaded JSON metadata: {str(e)}")

    combo_meta = {**embedded_meta, **external_meta}
    has_meta = len(combo_meta) > 0

    st.markdown("---")
    st.subheader("2. Reconstruction Mode & Parameter Settings")

    st.sidebar.header("⚙️ Phase Retrieval Controls")
    selected_method_name = st.sidebar.selectbox("Phase Retrieval Algorithm", options=list(RECON_METHODS.keys()), index=0)
    recon_method = RECON_METHODS[selected_method_name]

    if recon_method in ["griffin_lim", "spsi_gl"]:
        slider_label = "Griffin-Lim Iterations" if recon_method == "griffin_lim" else "Hybrid (SPSI+GL) Iterations"
        n_iter = st.sidebar.slider(slider_label, 10, 100, 50, 10)
    else:
        st.sidebar.caption(f"ℹ️ Iterations do not apply to {selected_method_name} (1-pass algorithm).")
        n_iter = 1

    force_manual = st.checkbox("Override with Manual Parameters (Mode B)", value=not has_meta)

    manual_params = {
        "force_manual": force_manual,
        "recon_method": recon_method,
        "n_iter": n_iter
    }

    if has_meta and not force_manual:
        st.success("✅ **Mode A Active**: Using detected metadata parameters.")
        st.json(combo_meta)
    else:
        st.warning("⚙️ **Mode B Active**: Enter parameter overrides manually below.")
        
        inferred_n_fft = 2 * (image_info['height'] - 1)
        
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            sr_input = st.number_input("Sample Rate (Hz)", min_value=4000, max_value=96000, value=int(combo_meta.get("sample_rate", 22050)), step=1000)
            scale_type_input = st.selectbox("Scaling Type", options=["log1p", "db"], index=0 if combo_meta.get("scale_type", "log1p") == "log1p" else 1)
        with col_p2:
            n_fft_input = st.number_input("FFT Size (n_fft)", min_value=64, max_value=8192, value=int(combo_meta.get("n_fft", inferred_n_fft)), step=128)
            hop_length_input = st.number_input("Hop Length", min_value=16, max_value=4096, value=int(combo_meta.get("hop_length", inferred_n_fft // 4)), step=32)
        with col_p3:
            scale_min_input = st.number_input("Scale Min", value=float(combo_meta.get("scaleMin", combo_meta.get("scale_min", 0.0))))
            scale_max_input = st.number_input("Scale Max", value=float(combo_meta.get("scaleMax", combo_meta.get("scale_max", 10.0))))

        manual_params.update({
            "sample_rate": sr_input,
            "n_fft": n_fft_input,
            "hop_length": hop_length_input,
            "scale_type": scale_type_input,
            "scale_min": scale_min_input,
            "scale_max": scale_max_input
        })

    # Trigger PNG to WAV Reconstruction
    if st.button("▶️ Synthesize WAV from PNG Spectrogram", type="primary"):
        with st.spinner("De-quantizing spectrogram and reconstructing phase..."):
            try:
                png_stream_recon = io.BytesIO(bytes_data)
                y_recon, sr, recovered_mag, result_meta = reconstruct_audio_from_png(
                    image_input=png_stream_recon,
                    metadata_input=combo_meta if has_meta else None,
                    manual_params=manual_params
                )
                st.session_state["png2wav_y_recon"] = y_recon
                st.session_state["png2wav_sr"] = sr
                st.session_state["png2wav_recovered_mag"] = recovered_mag
                st.session_state["png2wav_result_meta"] = result_meta
                st.session_state["png2wav_uint8_arr"] = uint8_arr
                st.success("Reconstruction complete!")
            except Exception as e:
                st.error(f"❌ Reconstruction Failed: {str(e)}")

    # Display Reconstruction Results
    if "png2wav_y_recon" in st.session_state:
        y_recon = st.session_state["png2wav_y_recon"]
        sr = st.session_state["png2wav_sr"]
        recovered_mag = st.session_state["png2wav_recovered_mag"]
        result_meta = st.session_state["png2wav_result_meta"]
        uint8_arr_disp = st.session_state["png2wav_uint8_arr"]

        st.markdown("---")
        st.subheader("3. Intermediate Visualizations & Synthesized Audio")
        
        tab_v1, tab_v2, tab_v3 = st.tabs([
            "1. Uploaded PNG Spectrogram Image",
            "2. Recovered Magnitude Spectrogram",
            "3. Reconstructed Waveform"
        ])

        with tab_v1:
            fig_img = plot_spectrogram(uint8_arr_disp, sr, result_meta["hop_length"], title="Uploaded PNG Image [0-255]", is_uint8=True)
            st.pyplot(fig_img)

        with tab_v2:
            fig_mag = plot_spectrogram(recovered_mag, sr, result_meta["hop_length"], title="De-quantized Magnitude Spectrogram", cmap="viridis")
            st.pyplot(fig_mag)

        with tab_v3:
            fig_wave = plot_waveform(y_recon, sr, title="Synthesized Audio Waveform")
            st.pyplot(fig_wave)

        render_audio_player(y_recon, sr, label="Reconstructed Audio Playback")

        wav_bytes = get_audio_download_bytes(y_recon, sr)
        st.download_button(
            label="📥 Download Reconstructed WAV File",
            data=wav_bytes,
            file_name="reconstructed_from_png.wav",
            mime="audio/wav"
        )

        st.markdown("---")
        st.subheader("4. Optional Original WAV Benchmark Comparison")
        st.markdown("Upload the original reference `.wav` file to measure objective reconstruction fidelity.")
        
        uploaded_orig_wav = st.file_uploader("Upload Original Reference WAV (.wav)", type=["wav"])
        if uploaded_orig_wav is not None:
            try:
                temp_ref_path = os.path.join("./data/output", "temp_orig_ref.wav")
                with open(temp_ref_path, "wb") as f:
                    f.write(uploaded_orig_wav.getbuffer())
                    
                y_ref, ref_sr, _ = load_audio(temp_ref_path, target_sr=sr)
                
                try:
                    snr_v = calculate_snr(y_ref, y_recon)
                except Exception:
                    snr_v = 0.0
                try:
                    stoi_v = calculate_stoi(y_ref, y_recon, sr)
                except Exception:
                    stoi_v = 0.0
                try:
                    pesq_v = calculate_pesq(y_ref, y_recon, sr)
                except Exception:
                    pesq_v = 0.0

                st.markdown("### 📊 Reconstruction Quality Metrics")
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("SNR (Signal-to-Noise Ratio)", f"{round(snr_v, 2)} dB")
                col_m2.metric("PESQ MOS Score", f"{round(pesq_v, 2)}")
                col_m3.metric("STOI Intelligibility", f"{round(stoi_v, 4)}")
            except Exception as e:
                st.error(f"Error calculating comparison metrics: {str(e)}")
