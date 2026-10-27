import numpy as np
import librosa
import scipy.signal
from typing import Optional, Literal
from .stft import compute_stft, compute_istft

def spsi_phase_inference(
    msgram: np.ndarray,
    n_fft: int,
    hop_length: int
) -> np.ndarray:
    """
    Single-pass Single-sideband Phase Inference (SPSI) algorithm.
    Constructs estimated phase matrix from magnitude spectral peaks and returns synthesized audio.
    
    Parameters:
        msgram: Magnitude spectrogram [num_bins, num_frames]
        n_fft: FFT size in samples
        hop_length: Hop length in samples
        
    Returns:
        1D reconstructed audio signal
    """
    num_bins, num_frames = msgram.shape
    y_out = np.zeros(num_frames * hop_length + n_fft - hop_length, dtype=np.float32)
    m_phase = np.zeros(num_bins, dtype=np.float32)
    m_win = scipy.signal.windows.hann(n_fft, sym=True).astype(np.float32)
    
    for i in range(num_frames):
        m_mag = msgram[:, i]
        for j in range(1, num_bins - 1):
            if (m_mag[j] > m_mag[j - 1]) and (m_mag[j] > m_mag[j + 1]):
                alpha = float(m_mag[j - 1])
                beta = float(m_mag[j])
                gamma = float(m_mag[j + 1])
                denom = alpha - 2.0 * beta + gamma
                
                if abs(denom) > 1e-12:
                    p = 0.5 * (alpha - gamma) / denom
                else:
                    p = 0.0
                    
                phase_rate = 2.0 * np.pi * (j + p) / n_fft
                m_phase[j] += hop_length * phase_rate
                peak_phase = m_phase[j]
                
                if p > 0:
                    bin_idx = j + 1
                    if bin_idx < num_bins:
                        m_phase[bin_idx] = peak_phase + np.pi
                    bin_idx = j - 1
                    while (bin_idx > 1) and (m_mag[bin_idx] < m_mag[bin_idx + 1]):
                        m_phase[bin_idx] = peak_phase + np.pi
                        bin_idx -= 1
                    bin_idx = j + 2
                    while (bin_idx < num_bins) and (m_mag[bin_idx] < m_mag[bin_idx - 1]):
                        m_phase[bin_idx] = peak_phase
                        bin_idx += 1
                        
                if p < 0:
                    bin_idx = j - 1
                    if bin_idx >= 0:
                        m_phase[bin_idx] = peak_phase + np.pi
                    bin_idx = j + 1
                    while (bin_idx < num_bins) and (m_mag[bin_idx] < m_mag[bin_idx - 1]):
                        m_phase[bin_idx] = peak_phase + np.pi
                        bin_idx += 1
                    bin_idx = j - 2
                    while (bin_idx > 1) and (m_mag[bin_idx] < m_mag[bin_idx + 1]):
                        m_phase[bin_idx] = peak_phase
                        bin_idx -= 1
                        
        magphase = m_mag * np.exp(1j * m_phase)
        magphase[0] = 0
        magphase[num_bins - 1] = 0
        
        m_recon_complex = np.concatenate([magphase, np.flip(np.conjugate(magphase[1:num_bins - 1]), 0)])
        m_recon = np.real(np.fft.ifft(m_recon_complex)) * m_win
        
        start_idx = i * hop_length
        y_out[start_idx : start_idx + n_fft] += m_recon.astype(np.float32)
        
    return y_out

def reconstruct_audio(
    msgram: np.ndarray,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: Optional[int] = None,
    window: str = "hann",
    center: bool = True,
    method: Literal["griffin_lim", "spsi", "spsi_gl", "random_phase"] = "griffin_lim",
    n_iter: int = 50,
    target_length: Optional[int] = None
) -> np.ndarray:
    """
    Reconstruct audio signal from a 2D magnitude spectrogram using specified phase retrieval method.
    
    Parameters:
        msgram: Magnitude spectrogram matrix [num_bins, num_frames]
        n_fft: FFT size in samples
        hop_length: Frame hop size
        win_length: Window length
        window: Window type
        center: Centering flag
        method: Phase retrieval method ('griffin_lim', 'spsi', 'spsi_gl', 'random_phase')
        n_iter: Number of Griffin-Lim iterations
        target_length: Desired length of audio output in samples
        
    Returns:
        Reconstructed audio array (float32)
    """
    if win_length is None:
        win_length = n_fft

    # Ensure magnitude is positive real values
    S_mag = np.maximum(msgram, 0.0).astype(np.float32)
    
    if method == "griffin_lim":
        y_rec = librosa.griffinlim(
            S=S_mag,
            n_iter=n_iter,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            length=None
        )
    elif method == "spsi":
        y_rec = spsi_phase_inference(S_mag, n_fft=win_length, hop_length=hop_length)
    elif method == "spsi_gl":
        # 1. Initial SPSI estimate
        y_init = spsi_phase_inference(S_mag, n_fft=win_length, hop_length=hop_length)
        # 2. Get initial phase from SPSI STFT
        init_stft = compute_stft(y_init, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, center=center)
        p = np.angle(init_stft)
        
        # Match dimensions if needed
        min_frames = min(S_mag.shape[1], p.shape[1])
        S_sub = S_mag[:, :min_frames]
        p_sub = p[:, :min_frames]
        
        y_rec = y_init
        for _ in range(n_iter):
            S_complex = S_sub * np.exp(1j * p_sub)
            y_rec = compute_istft(S_complex, hop_length=hop_length, win_length=win_length, window=window, center=center, length=None)
            stft_next = compute_stft(y_rec, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, center=center)
            p_sub = np.angle(stft_next[:, :min_frames])
    elif method == "random_phase":
        random_phase = np.exp(1j * np.random.uniform(-np.pi, np.pi, size=S_mag.shape))
        S_complex = S_mag * random_phase
        y_rec = compute_istft(S_complex, hop_length=hop_length, win_length=win_length, window=window, center=center, length=None)
    else:
        raise ValueError(f"Unknown reconstruction method: {method}")

    # Truncate or pad y_rec to target_length if specified
    if target_length is not None:
        if len(y_rec) > target_length:
            y_rec = y_rec[:target_length]
        elif len(y_rec) < target_length:
            y_rec = np.pad(y_rec, (0, target_length - len(y_rec)))

    # Peak normalization safety check
    max_amp = np.max(np.abs(y_rec))
    if max_amp > 1.0:
        y_rec = y_rec / max_amp * 0.99
        
    return y_rec.astype(np.float32)
