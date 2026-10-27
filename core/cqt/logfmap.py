import numpy as np
from typing import Tuple

def compute_logfmap(
    I: int,
    L: int = 10,
    H: int = 513
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Construct forward mapping matrix M and recovery matrix N for Linear to Log-Frequency (CQT) conversion.
    Ported and updated from Dan Ellis' MATLAB logfmap function.
    
    Parameters:
        I: Number of frequency rows in original linear spectrogram (n_fft // 2 + 1)
        L: Lowest FFT bin to preserve (> 0)
        H: Highest FFT bin to preserve (<= I)
        
    Returns:
        M: Mapping matrix of shape (opr, I)
        N: Recovery matrix of shape (I, opr)
        opr: Number of log-frequency bins
    """
    if L < 1:
        L = 1
    if H > I:
        H = I
        
    ratio = (H - 1.0) / float(H)
    if ratio <= 0 or ratio >= 1:
        ratio = 0.999
        
    # Calculate number of log frequency bins
    opr = int(np.round(np.log(L / float(H)) / np.log(ratio)))
    if opr < 1:
        opr = 1
        
    # Fractional bin indices
    ibin = L * np.exp(np.arange(opr, dtype=np.float64) * (-np.log(ratio)))
    
    M = np.zeros((opr, I), dtype=np.float32)
    eps = np.finfo(np.float32).eps
    
    for i in range(opr):
        tt = np.pi * (np.arange(I, dtype=np.float64) - ibin[i])
        M[i, :] = ((np.sin(tt) + eps) / (tt + eps)).astype(np.float32)
        
    # Gain vector normalization
    G = np.ones(I, dtype=np.float32)
    G[:H] = np.arange(H, dtype=np.float32) / float(H)
    
    G_mat = np.tile(G, (opr, 1))
    N = np.transpose(M * G_mat).astype(np.float32)
    
    return M, N, opr
