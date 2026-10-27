import numpy as np
from typing import Tuple, Dict, Any, Optional, Literal
from ..audio.stft import compute_stft
from .scaling import magnitude_to_scaled
from .quantization import quantize_to_8bit

def generate_spectrogram(
    y: np.ndarray,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: Optional[int] = None,
    window: str = "hann",
    center: bool = True,
    scale_type: Literal["log1p", "db"] = "log1p",
    scale_min: Optional[float] = None,
    scale_max: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Generate linear magnitude, scaled, and 8-bit quantized spectrograms from audio array.
    
    Returns:
        stft_complex: Complex STFT matrix [num_bins, num_frames]
        mag_sgram: Linear magnitude spectrogram [num_bins, num_frames]
        scaled_sgram: Log/dB scaled magnitude spectrogram [num_bins, num_frames]
        quant_img: 8-bit uint8 image array [num_bins, num_frames]
        metadata: Dictionary containing STFT parameters and quantization metadata
    """
    stft_complex = compute_stft(
        y=y,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center
    )
    
    mag_sgram = np.abs(stft_complex).astype(np.float32)
    scaled_sgram = magnitude_to_scaled(mag_sgram, scale_type=scale_type)
    quant_img, q_meta = quantize_to_8bit(scaled_sgram, scale_min=scale_min, scale_max=scale_max)
    
    num_bins, num_frames = mag_sgram.shape
    metadata = {
        "n_fft": n_fft,
        "hop_length": hop_length,
        "win_length": win_length if win_length is not None else n_fft,
        "window": window,
        "center": center,
        "scale_type": scale_type,
        "num_bins": num_bins,
        "num_frames": num_frames,
        **q_meta
    }
    
    return stft_complex, mag_sgram, scaled_sgram, quant_img, metadata
