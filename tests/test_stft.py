import numpy as np
import pytest
from core.audio.stft import compute_stft, compute_istft

def test_stft_istft_roundtrip():
    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Generate 440 Hz sine wave signal
    y = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    n_fft = 1024
    hop_length = 256
    
    stft_mat = compute_stft(y, n_fft=n_fft, hop_length=hop_length)
    assert stft_mat.shape[0] == n_fft // 2 + 1
    
    y_rec = compute_istft(stft_mat, hop_length=hop_length, length=len(y))
    assert len(y_rec) == len(y)
    
    # Check reconstruction fidelity when exact complex STFT (magnitude + phase) is preserved
    mse = np.mean((y - y_rec) ** 2)
    assert mse < 1e-4

if __name__ == "__main__":
    test_stft_istft_roundtrip()
    print("test_stft_istft_roundtrip passed!")
