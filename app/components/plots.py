import matplotlib.pyplot as plt
import numpy as np
import librosa.display
from typing import Optional

def plot_waveform(y: np.ndarray, sr: int, title: str = "Audio Waveform"):
    """
    Generate waveform plot figure.
    """
    fig, ax = plt.subplots(figsize=(10, 2.5), dpi=100)
    times = np.linspace(0, len(y) / sr, len(y))
    ax.plot(times, y, color="#1f77b4", alpha=0.8, linewidth=0.8)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("Time (seconds)", fontsize=10)
    ax.set_ylabel("Amplitude", fontsize=10)
    ax.set_xlim(0, max(times) if len(times) > 0 else 1)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    return fig

def plot_spectrogram(
    sgram: np.ndarray,
    sr: int,
    hop_length: int,
    title: str = "Spectrogram",
    ylabel: str = "Frequency (Hz)",
    cmap: str = "viridis",
    is_uint8: bool = False
):
    """
    Generate 2D spectrogram heatmap plot figure.
    """
    fig, ax = plt.subplots(figsize=(10, 4), dpi=100)
    
    num_bins, num_frames = sgram.shape
    times = np.arange(num_frames) * (hop_length / sr)
    freqs = np.linspace(0, sr / 2, num_bins)
    
    if is_uint8:
        # uint8 array is vertically flipped for display (0 Hz at bottom)
        img_display = np.flipud(sgram)
        im = ax.imshow(
            img_display,
            aspect="auto",
            extent=[0, max(times) if len(times) > 0 else 1, 0, sr / 2],
            cmap="gray",
            vmin=0,
            vmax=255
        )
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("8-bit Intensity [0-255]", fontsize=10)
    else:
        im = ax.imshow(
            sgram,
            origin="lower",
            aspect="auto",
            extent=[0, max(times) if len(times) > 0 else 1, 0, sr / 2],
            cmap=cmap
        )
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Magnitude / Scaled dB", fontsize=10)
        
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("Time (seconds)", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    plt.tight_layout()
    return fig

def plot_spectrogram_comparison(
    sgram1: np.ndarray,
    sgram2: np.ndarray,
    sr: int,
    hop_length: int,
    title1: str = "Original",
    title2: str = "Reconstructed",
    cmap: str = "viridis"
):
    """
    Generate side-by-side comparison figure for two spectrograms.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), dpi=100)
    
    num_bins, num_frames = sgram1.shape
    max_time = num_frames * (hop_length / sr)
    
    im1 = ax1.imshow(sgram1, origin="lower", aspect="auto", extent=[0, max_time, 0, sr / 2], cmap=cmap)
    ax1.set_title(title1, fontsize=11, fontweight="bold")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Frequency (Hz)")
    plt.colorbar(im1, ax=ax1)
    
    im2 = ax2.imshow(sgram2, origin="lower", aspect="auto", extent=[0, max_time, 0, sr / 2], cmap=cmap)
    ax2.set_title(title2, fontsize=11, fontweight="bold")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Frequency (Hz)")
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    return fig

def plot_difference_heatmap(
    diff_sgram: np.ndarray,
    sr: int,
    hop_length: int,
    title: str = "Spectrogram Difference / Error Heatmap"
):
    """
    Generate absolute difference heatmap plot.
    """
    fig, ax = plt.subplots(figsize=(10, 4), dpi=100)
    num_bins, num_frames = diff_sgram.shape
    max_time = num_frames * (hop_length / sr)
    
    im = ax.imshow(diff_sgram, origin="lower", aspect="auto", extent=[0, max_time, 0, sr / 2], cmap="magma")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Absolute Magnitude Difference", fontsize=10)
    
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("Time (seconds)", fontsize=10)
    ax.set_ylabel("Frequency (Hz)", fontsize=10)
    plt.tight_layout()
    return fig
