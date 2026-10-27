import numpy as np
import scipy.signal
from typing import Tuple, Dict, Any, Optional
from .logfmap import compute_logfmap

def linear_to_cqt(
    linear_sgram: np.ndarray,
    n_fft: int = 1024,
    low_row: int = 10,
    cqt_bins: Optional[int] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Convert a linear frequency magnitude/scaled spectrogram into a Constant-Q (CQT) spectrogram.
    
    Parameters:
        linear_sgram: Linear frequency spectrogram matrix [num_bins, num_frames]
        n_fft: Original FFT window size
        low_row: Lowest frequency bin to include
        cqt_bins: Number of output frequency bins (defaults to num_bins)
        
    Returns:
        cqt_sgram: CQT spectrogram matrix [cqt_bins, num_frames]
        metadata: CQT transformation parameters
    """
    num_bins, num_frames = linear_sgram.shape
    if cqt_bins is None:
        cqt_bins = num_bins
        
    M, N, opr = compute_logfmap(I=num_bins, L=low_row, H=num_bins)
    
    # Perform matrix multiplication: M (opr, num_bins) x D (num_bins, num_frames) -> (opr, num_frames)
    MD = np.dot(M, linear_sgram)
    
    # Resample along frequency axis if cqt_bins differs from opr
    if cqt_bins != opr:
        MD_resampled = scipy.signal.resample(MD, cqt_bins, axis=0)
    else:
        MD_resampled = MD
        
    cqt_sgram = np.maximum(MD_resampled, 0.0).astype(np.float32)
    
    metadata = {
        "linFreqBins": num_bins,
        "lowRow": low_row,
        "logFreqBins": cqt_bins,
        "oprBins": opr,
        "n_fft": n_fft
    }
    
    return cqt_sgram, metadata
