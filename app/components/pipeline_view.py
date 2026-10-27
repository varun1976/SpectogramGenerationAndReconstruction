import streamlit as st

def render_pipeline_diagram():
    """
    Render HTML/Markdown pipeline diagram.
    """
    pipeline_markdown = """
    ```text
    ┌──────────────┐     ┌──────────────┐     ┌──────────────────────┐
    │  WAV Audio   │ ──> │  STFT / FFT  │ ──> │ Magnitude Spectrogram│
    └──────────────┘     └──────────────┘     └──────────────────────┘
                                                         │
                                                         ▼
    ┌──────────────┐     ┌──────────────┐     ┌──────────────────────┐
    │  PNG / JPEG  │ <── │ 8-bit Quant  │ <── │ Decibel / Log Scale  │
    └──────────────┘     └──────────────┘     └──────────────────────┘
           │
           ├──> Pipeline A: De-quantize ──> Phase Retrieval (GL/SPSI) ──> ISTFT ──> Recon WAV
           │
           ├──> Pipeline B: JPEG (100/80/50) ──> De-quantize ──> Phase Retrieval ──> Recon WAV
           │
           └──> Pipeline C: Linear -> CQT -> CQT Image -> CQT -> Linear ──> Recon WAV
    ```
    """
    st.markdown(pipeline_markdown)
