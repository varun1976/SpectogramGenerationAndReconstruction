import numpy as np
from core.audio.stft import compute_stft
from core.audio.reconstruction import reconstruct_audio

def test_phase_retrieval_methods():
    sr = 16000
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
    y = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    stft_mat = compute_stft(y, n_fft=512, hop_length=128)
    msgram = np.abs(stft_mat)
    
    methods = ["griffin_lim", "spsi", "spsi_gl", "random_phase"]
    for method in methods:
        y_rec = reconstruct_audio(
            msgram=msgram,
            n_fft=512,
            hop_length=128,
            method=method,
            n_iter=10,
            target_length=len(y)
        )
        assert len(y_rec) == len(y)
        assert not np.isnan(y_rec).any()
        assert np.max(np.abs(y_rec)) > 0.0

if __name__ == "__main__":
    test_phase_retrieval_methods()
    print("test_phase_retrieval_methods passed!")
