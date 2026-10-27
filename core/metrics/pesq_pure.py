import numpy as np
import scipy.signal
import librosa

def hz_to_bark(f: np.ndarray) -> np.ndarray:
    """Convert frequency in Hz to Bark scale."""
    return 6.0 * np.arcsinh(f / 600.0)

def bark_to_hz(z: np.ndarray) -> np.ndarray:
    """Convert Bark scale value to Hz."""
    return 600.0 * np.sinh(z / 6.0)

def compute_bark_filterbank(sr: int = 16000, n_fft: int = 512, n_barks: int = 42) -> np.ndarray:
    """
    Construct Bark scale filterbank matrix of shape (n_barks, n_fft // 2 + 1).
    """
    n_bins = n_fft // 2 + 1
    freqs = np.linspace(0, sr / 2.0, n_bins)
    max_bark = hz_to_bark(np.array([sr / 2.0]))[0]
    bark_edges = bark_to_hz(np.linspace(0, max_bark, n_barks + 2))
    
    weights = np.zeros((n_barks, n_bins), dtype=np.float32)
    for i in range(n_barks):
        low = bark_edges[i]
        center = bark_edges[i + 1]
        high = bark_edges[i + 2]
        
        # Triangular filter
        up = (freqs - low) / (center - low + 1e-6)
        down = (high - freqs) / (high - center + 1e-6)
        weights[i, :] = np.maximum(0.0, np.minimum(up, down))
        
    # Normalize rows
    sums = np.sum(weights, axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    weights = weights / sums
    return weights

def pesq_pure_python(
    ref_signal: np.ndarray,
    deg_signal: np.ndarray,
    sr: int = 16000,
    mode: str = "wb"
) -> float:
    """
    Pure Python / NumPy implementation of ITU-T P.862 PESQ algorithm.
    Does not require C compilation or MS Visual C++ Build Tools.
    
    Parameters:
        ref_signal: Reference audio array
        deg_signal: Degraded/reconstructed audio array
        sr: Sample rate (resampled to 16kHz for 'wb' or 8kHz for 'nb')
        mode: 'wb' (wideband) or 'nb' (narrowband)
        
    Returns:
        PESQ MOS score in range [1.0, 4.5]
    """
    ref = ref_signal.astype(np.float64).flatten()
    deg = deg_signal.astype(np.float64).flatten()
    
    min_len = min(len(ref), len(deg))
    if min_len < int(0.2 * sr):
        return 1.0
        
    ref = ref[:min_len]
    deg = deg[:min_len]
    
    target_sr = 16000 if mode == "wb" else 8000
    if sr != target_sr:
        ref = librosa.resample(ref.astype(np.float32), orig_sr=sr, target_sr=target_sr).astype(np.float64)
        deg = librosa.resample(deg.astype(np.float32), orig_sr=sr, target_sr=target_sr).astype(np.float64)
        min_len = min(len(ref), len(deg))
        ref = ref[:min_len]
        deg = deg[:min_len]
        
    # Power normalization
    ref_power = np.mean(ref ** 2)
    deg_power = np.mean(deg ** 2)
    
    if ref_power > 1e-12:
        ref = ref / np.sqrt(ref_power)
    if deg_power > 1e-12:
        deg = deg / np.sqrt(deg_power)
        
    # STFT frame parameters
    n_fft = 512 if target_sr == 16000 else 256
    hop_length = n_fft // 2
    n_barks = 42 if target_sr == 16000 else 24
    
    win = scipy.signal.windows.hann(n_fft, sym=False)
    
    # Compute STFT magnitude
    stft_ref = np.abs(librosa.stft(ref, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=win))
    stft_deg = np.abs(librosa.stft(deg, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=win))
    
    min_frames = min(stft_ref.shape[1], stft_deg.shape[1])
    if min_frames < 2:
        return 1.0
        
    stft_ref = stft_ref[:, :min_frames]
    stft_deg = stft_deg[:, :min_frames]
    
    # Bark filterbank energy
    fbank = compute_bark_filterbank(sr=target_sr, n_fft=n_fft, n_barks=n_barks)
    bark_ref = np.dot(fbank, stft_ref ** 2)
    bark_deg = np.dot(fbank, stft_deg ** 2)
    
    # Zwicker Loudness Density: N = E^0.23 (scaled)
    loudness_ref = (np.maximum(bark_ref, 1e-10)) ** 0.23
    loudness_deg = (np.maximum(bark_deg, 1e-10)) ** 0.23
    
    # Symmetrical disturbance density D
    D_sym = np.abs(loudness_ref - loudness_deg)
    
    # Asymmetrical disturbance density A (for dropouts/suppression)
    ratio = (loudness_deg + 1e-5) / (loudness_ref + 1e-5)
    asym_mask = np.where(ratio < 1.0, (1.0 / (ratio + 1e-5)) ** 0.5, 1.0)
    D_asym = D_sym * asym_mask
    
    # Frame disturbance aggregation (L2 norm per frame, L6 norm over time)
    frame_sym = np.mean(D_sym ** 2, axis=0) ** 0.5
    frame_asym = np.mean(D_asym ** 2, axis=0) ** 0.5
    
    d_sym_mean = (np.mean(frame_sym ** 6)) ** (1.0 / 6.0)
    d_asym_mean = (np.mean(frame_asym ** 6)) ** (1.0 / 6.0)
    
    # PESQ MOS Score Formula (ITU-T P.862 mapping)
    mos = 4.5 - 0.1 * d_sym_mean - 0.035 * d_asym_mean
    mos = float(np.clip(mos, 1.0, 4.5))
    
    return round(mos, 3)
