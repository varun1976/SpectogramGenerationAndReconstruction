import numpy as np
import scipy.signal
from typing import Dict, Any, Optional
from .logfmap import compute_logfmap

def cqt_to_linear(
    cqt_sgram: np.ndarray,
    target_bins: int = 513,
    low_row: int = 10,
    metadata: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Convert a Constant-Q (CQT) spectrogram back into a linear frequency spectrogram.
    
    Parameters:
        cqt_sgram: CQT spectrogram matrix [cqt_bins, num_frames]
        target_bins: Target number of linear frequency bins (n_fft // 2 + 1)
        low_row: Lowest frequency bin
        metadata: Optional metadata dictionary containing linFreqBins/lowRow
        
    Returns:
        Reconstructed linear frequency spectrogram matrix [target_bins, num_frames]
    """
    if metadata is not None:
        target_bins = metadata.get("linFreqBins", target_bins)
        low_row = metadata.get("lowRow", low_row)
        
    M, N, opr = compute_logfmap(I=target_bins, L=low_row, H=target_bins)
    
    curr_bins, num_frames = cqt_sgram.shape
    if curr_bins != opr:
        # Upsample or downsample along frequency axis to match opr
        cqt_resampled = scipy.signal.resample(cqt_sgram, opr, axis=0)
    else:
        cqt_resampled = cqt_sgram
        
    # Perform matrix multiplication: N (target_bins, opr) x CQT (opr, num_frames) -> (target_bins, num_frames)
    ND = np.dot(N, cqt_resampled)
    
    recon_linear = np.maximum(ND, 0.0).astype(np.float32)
    return recon_linear
