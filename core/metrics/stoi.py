import numpy as np

def calculate_stoi(
    ref_signal: np.ndarray,
    deg_signal: np.ndarray,
    sr: int,
    extended: bool = False
) -> float:
    """
    Calculate Short-Time Objective Intelligibility (STOI) score.
    Safely handles missing pystoi package or runtime failures.
    
    Parameters:
        ref_signal: Reference audio signal
        deg_signal: Degraded/reconstructed audio signal
        sr: Sample rate in Hz
        extended: Whether to use extended STOI (eSTOI)
        
    Returns:
        STOI score (float in range 0.0 to 1.0, or 0.0 if pystoi unavailable)
    """
    try:
        try:
            import pystoi
        except ImportError:
            return 0.0
            
        ref = ref_signal.astype(np.float64).flatten()
        deg = deg_signal.astype(np.float64).flatten()
        
        min_len = min(len(ref), len(deg))
        if min_len < int(0.1 * sr):
            return 0.0
            
        ref = ref[:min_len]
        deg = deg[:min_len]
        
        score = pystoi.stoi(ref, deg, sr, extended=extended)
        return float(score)
    except Exception:
        return 0.0
