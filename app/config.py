import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
INPUT_DIR = os.path.join(DATA_DIR, "input")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Default Processing Configuration
DEFAULT_CONFIG = {
    "sample_rate": 22050,
    "n_fft": 1024,
    "hop_length": 256,
    "win_length": 1024,
    "window": "hann",
    "scale_type": "log1p",
    "cqt_low_row": 10,
    "cqt_bins": 513,
    "recon_method": "griffin_lim",
    "gl_iterations": 50
}

# Supported Windows Functions
WINDOW_FUNCTIONS = ["hann", "hamming", "blackman", "boxcar"]

# Supported Phase Retrieval Methods
RECON_METHODS = {
    "Griffin-Lim": "griffin_lim",
    "SPSI (Single-pass Single-sideband)": "spsi",
    "Hybrid SPSI + Griffin-Lim": "spsi_gl",
    "Random Phase Baseline": "random_phase"
}

# Color Palettes
COLORMAP_SPECTROGRAM = "viridis"
COLORMAP_DIFF = "magma"
COLORMAP_CQT = "inferno"
