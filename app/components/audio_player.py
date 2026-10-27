import io
import streamlit as st
import soundfile as sf
import numpy as np

def render_audio_player(y: np.ndarray, sr: int, label: str = "Audio Player"):
    """
    Render Streamlit audio player for a float numpy array.
    """
    st.markdown(f"**{label}**")
    st.audio(y, sample_rate=sr)

def get_audio_download_bytes(y: np.ndarray, sr: int) -> bytes:
    """
    Convert audio float array into WAV file bytes buffer.
    """
    buffer = io.BytesIO()
    sf.write(buffer, y, sr, format='WAV', subtype='PCM_16')
    buffer.seek(0)
    return buffer.read()
