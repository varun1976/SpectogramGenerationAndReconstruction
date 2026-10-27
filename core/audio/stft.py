import numpy as np
import librosa
from typing import Optional, Union

def compute_stft(
    y: np.ndarray,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: Optional[int] = None,
    window: str = "hann",
    center: bool = True
) -> np.ndarray:
    """
    Compute Short-Time Fourier Transform (STFT) of 1D audio signal.
    
    Parameters:
        y: Audio signal array (1D float32)
        n_fft: FFT window size in samples
        hop_length: Frame hop size in samples
        win_length: Window length (defaults to n_fft)
        window: Window function type ('hann', 'hamming', 'blackman', etc.)
        center: Centering flag for boundary padding
        
    Returns:
        Complex STFT matrix of shape (1 + n_fft // 2, n_frames)
    """
    if win_length is None:
        win_length = n_fft
        
    stft_matrix = librosa.stft(
        y=y,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center
    )
    return stft_matrix

def compute_istft(
    stft_matrix: np.ndarray,
    hop_length: int = 256,
    win_length: Optional[int] = None,
    window: str = "hann",
    center: bool = True,
    length: Optional[int] = None
) -> np.ndarray:
    """
    Compute Inverse Short-Time Fourier Transform (ISTFT).
    
    Parameters:
        stft_matrix: Complex STFT matrix
        hop_length: Hop length
        win_length: Window length
        window: Window type
        center: Centering flag matching forward STFT
        length: Expected output audio length in samples
        
    Returns:
        1D reconstructed audio signal (float32 array)
    """
    n_fft = 2 * (stft_matrix.shape[0] - 1)
    if win_length is None:
        win_length = n_fft
        
    y_rec = librosa.istft(
        stft_matrix,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        length=length
    )
    return y_rec.astype(np.float32)
