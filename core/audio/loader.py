import os
import numpy as np
import librosa
import soundfile as sf
from typing import Tuple, Dict, Any, Optional

def load_audio(
    file_path: str,
    target_sr: Optional[int] = None,
    mono: bool = True,
    duration: Optional[float] = None
) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    """
    Load an audio file into a NumPy floating point array.
    
    Parameters:
        file_path: Path to the input audio file (.wav, etc.)
        target_sr: Optional target sample rate. If None, uses native sample rate.
        mono: Whether to convert multi-channel audio to mono.
        duration: Optional maximum duration in seconds to load.
        
    Returns:
        y: Audio signal array (1D float32 array for mono).
        sr: Sample rate.
        metadata: Dictionary containing audio parameters.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
        
    try:
        # Load audio using librosa
        y, sr = librosa.load(file_path, sr=target_sr, mono=mono, duration=duration)
        y = y.astype(np.float32)
        
        n_samples = len(y)
        dur = n_samples / sr
        
        metadata = {
            "file_path": file_path,
            "filename": os.path.basename(file_path),
            "sample_rate": sr,
            "duration": dur,
            "n_samples": n_samples,
            "n_channels": 1 if y.ndim == 1 else y.shape[0],
            "max_amplitude": float(np.max(np.abs(y))) if n_samples > 0 else 0.0
        }
        return y, sr, metadata
    except Exception as e:
        raise RuntimeError(f"Error loading audio file '{file_path}': {str(e)}")

def save_audio(file_path: str, y: np.ndarray, sr: int, normalize: bool = True) -> str:
    """
    Save floating point audio array to a WAV file.
    
    Parameters:
        file_path: Target WAV file path.
        y: Audio signal array.
        sr: Sample rate.
        normalize: If True, normalizes peak amplitude to 0.99 if max amplitude > 1.0 to prevent clipping.
        
    Returns:
        Saved file path.
    """
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    
    signal = y.copy().astype(np.float32)
    max_val = np.max(np.abs(signal))
    
    if normalize and max_val > 1.0:
        signal = signal / max_val * 0.99
        
    sf.write(file_path, signal, sr, subtype='PCM_16')
    return file_path

def pad_or_truncate(y: np.ndarray, target_length: int) -> np.ndarray:
    """
    Pad with zeros or truncate signal to match exact target sample length.
    """
    if len(y) == target_length:
        return y
    elif len(y) > target_length:
        return y[:target_length]
    else:
        return np.pad(y, (0, target_length - len(y)), mode='constant')
