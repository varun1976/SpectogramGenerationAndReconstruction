import numpy as np

def calculate_snr(
    ref_signal: np.ndarray,
    deg_signal: np.ndarray,
    eps: float = 1e-10
) -> float:
    """
    Calculate Signal-to-Noise Ratio (SNR) in dB between reference and degraded audio signals.
    
    Parameters:
        ref_signal: Reference original audio array (1D float)
        deg_signal: Degraded/reconstructed audio array (1D float)
        eps: Small epsilon constant to prevent division by zero
        
    Returns:
        SNR score in decibels (dB)
    """
    ref = ref_signal.astype(np.float64).flatten()
    deg = deg_signal.astype(np.float64).flatten()
    
    # Align signal lengths
    min_len = min(len(ref), len(deg))
    if min_len == 0:
        return 0.0
        
    ref = ref[:min_len]
    deg = deg[:min_len]
    
    signal_power = np.sum(ref ** 2)
    noise_power = np.sum((ref - deg) ** 2)
    
    if noise_power < eps:
        return 100.0  # Perfect reconstruction cap
    if signal_power < eps:
        return 0.0
        
    snr_val = 10.0 * np.log10(signal_power / noise_power)
    return float(snr_val)
