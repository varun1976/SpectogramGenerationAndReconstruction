import numpy as np
import librosa
from typing import Literal

def magnitude_to_scaled(
    msgram: np.ndarray,
    scale_type: Literal["log1p", "db"] = "log1p",
    top_db: float = 80.0
) -> np.ndarray:
    """
    Convert linear magnitude spectrogram to log or decibel (dB) scale.
    
    Parameters:
        msgram: Linear magnitude spectrogram matrix
        scale_type: 'log1p' (np.log1p(x)) or 'db' (amplitude_to_db)
        top_db: Dynamic range cutoff for dB scale
        
    Returns:
        Scaled magnitude matrix (float32 array)
    """
    msgram_clean = np.maximum(msgram, 0.0)
    
    if scale_type == "log1p":
        return np.log1p(msgram_clean).astype(np.float32)
    elif scale_type == "db":
        db_spec = librosa.amplitude_to_db(msgram_clean, ref=1.0, top_db=top_db)
        return db_spec.astype(np.float32)
    else:
        raise ValueError(f"Unsupported scale_type: {scale_type}")

def scaled_to_magnitude(
    scaled_sgram: np.ndarray,
    scale_type: Literal["log1p", "db"] = "log1p"
) -> np.ndarray:
    """
    Convert log or dB scaled spectrogram back to linear magnitude.
    
    Parameters:
        scaled_sgram: Scaled magnitude matrix
        scale_type: 'log1p' or 'db'
        
    Returns:
        Linear magnitude matrix (float32 array)
    """
    if scale_type == "log1p":
        mag = np.expm1(scaled_sgram)
        return np.maximum(mag, 0.0).astype(np.float32)
    elif scale_type == "db":
        mag = librosa.db_to_amplitude(scaled_sgram, ref=1.0)
        return np.maximum(mag, 0.0).astype(np.float32)
    else:
        raise ValueError(f"Unsupported scale_type: {scale_type}")
