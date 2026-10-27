import sys
import os
import numpy as np

# Add project root to sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.audio.loader import load_audio
from core.audio.reconstruction import reconstruct_audio
from core.spectrogram.generator import generate_spectrogram
from core.metrics.pesq import calculate_pesq
from core.metrics.stoi import calculate_stoi
from core.metrics.snr import calculate_snr

def main():
    print("=" * 65)
    print("PESQ & Perceptual Metrics Diagnostic Tool")
    print("=" * 65)
    print(f"Python Executable : {sys.executable}")
    print(f"Python Version    : {sys.version.split()[0]}")
    print(f"Project Root      : {ROOT_DIR}")
    
    # Check if PyPI C-extension pesq is installed
    try:
        import pesq as pesq_c
        print(f"PyPI C-extension  : INSTALLED ({pesq_c.__file__})")
    except ImportError:
        print("PyPI C-extension  : NOT INSTALLED (using pure Python ITU-T P.862 engine)")
        
    sample_path = os.path.join(ROOT_DIR, "data", "input", "sample.wav")
    if not os.path.exists(sample_path):
        print(f"Sample WAV file missing at '{sample_path}'. Generating synthetic audio...")
        sr = 22050
        t = np.linspace(0, 2.0, sr * 2)
        y = (0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)
    else:
        y, sr, _ = load_audio(sample_path)
        
    print(f"\nAudio File Loaded : '{sample_path}'")
    print(f"Sample Rate       : {sr} Hz")
    print(f"Duration          : {len(y) / sr:.2f} s ({len(y)} samples)")
    
    # Run STFT & reconstruction
    _, mag_sgram, _, _, _ = generate_spectrogram(y=y, n_fft=1024, hop_length=256)
    y_recon = reconstruct_audio(msgram=mag_sgram, n_fft=1024, hop_length=256, method="griffin_lim", n_iter=30)
    
    snr_val = calculate_snr(y, y_recon)
    stoi_val = calculate_stoi(y, y_recon, sr)
    pesq_val = calculate_pesq(y, y_recon, sr)
    
    print("\n--- Diagnostic Metric Output ---")
    print(f"SNR  Score        : {snr_val:.2f} dB")
    print(f"STOI Score        : {stoi_val:.4f}")
    print(f"PESQ Score        : {pesq_val:.3f}")
    print("=" * 65)

if __name__ == "__main__":
    main()
