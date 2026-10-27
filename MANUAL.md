# User & Research Manual: Spectrogram Generation & Reconstruction Platform

Welcome to the **Spectrogram Generation and Reconstruction Platform**. This manual provides a step-by-step guide to using the web interface and operating the underlying Python signal-processing core.

---

## Table of Contents
1. [Platform Overview & Features](#1-platform-overview--features)
2. [Quick-Start & Installation](#2-quick-start--installation)
3. [Operating the Web Application](#3-operating-the-web-application)
   - [Page 1: Dashboard](#page-1-dashboard)
   - [Page 2: Spectrogram Generation](#page-2-spectrogram-generation)
   - [Page 3: Image Compression Experiment](#page-3-image-compression-experiment)
   - [Page 4: Constant-Q Transform (CQT) Experiment](#page-4-constant-q-transform-cqt-experiment)
   - [Page 5: Audio Reconstruction](#page-5-audio-reconstruction)
4. [Using the Python Core API Programmatically](#4-using-the-python-core-api-programmatically)
5. [Running Unit Tests](#5-running-unit-tests)
6. [Troubleshooting & FAQ](#6-troubleshooting--faq)

---

## 1. Platform Overview & Features

This research platform allows users to experimentally evaluate acoustic information loss introduced when transforming 1D audio signals into 2D spectrogram images and Constant-Q representations.

### Key Capabilities

- **Audio Loaders**: Supports `.wav` file uploads, multi-channel stereo-to-mono conversion, duration truncation, and synthetic audio generation.
- **STFT Parameter Controls**: Interactive sliders for `n_fft`, `hop_length`, `win_length`, `window_type` (`hann`, `hamming`, `blackman`, `boxcar`), and magnitude scaling (`log1p` or `db`).
- **8-Bit Quantization**: Maps continuous spectral magnitudes to 8-bit uint8 `[0, 255]` images with metadata preservation (PNG `PngInfo` text metadata and JPEG sidecar JSON files).
- **PNG & JPEG Compression Experiments**: Compares Lossless PNG against lossy JPEG formats (Quality 100%, 80%, 50%) with automatic SNR, PESQ, and STOI evaluation.
- **Constant-Q Transform (CQT) Engine**: Converts linear-frequency STFT spectrograms to logarithmic frequency space (`Linear -> CQT -> Linear`) using Dan Ellis' `logfmap` matrix algorithm.
- **Phase Retrieval Algorithms**:
  1. **Griffin-Lim**: Iterative magnitude projection phase estimation.
  2. **SPSI**: Single-pass Single-sideband Phase Inference algorithm.
  3. **Hybrid SPSI + Griffin-Lim**: Uses SPSI as phase initializer followed by Griffin-Lim iterations.
  4. **Random Phase Baseline**: Uniform random phase initialization.
- **Export Capabilities**: Audio playback directly in browser, WAV file downloads, PNG/JPEG image downloads, and CSV experiment logging.

---

## 2. Quick-Start & Installation

### Prerequisites
- **Python**: 3.11+
- **OS**: Windows, macOS, or Linux

### Launch Command

Open PowerShell or Terminal in the project directory and execute:

```powershell
python run.py
```

*Alternatively, launch Streamlit directly:*

```powershell
streamlit run app/main.py
```

The web application will open automatically in your browser at `http://localhost:8501`.

---

## 3. Operating the Web Application

The application features a sidebar navigation menu with 5 dedicated pages:

```text
┌─────────────────────────────────────────────────────────┐
│ 🎵 Navigation                                           │
│  ● 1. Dashboard                                         │
│  ○ 2. Spectrogram Generation                            │
│  ○ 3. Image Compression Experiment                      │
│  ○ 4. CQT Experiment                                    │
│  ○ 5. Audio Reconstruction                              │
└─────────────────────────────────────────────────────────┘
```

---

### Page 1: Dashboard

- **Purpose**: Provides an executive summary of research objectives, system architecture, and processing pipelines.
- **Features**:
  - Displays the 3 core research questions (Quantization Loss, JPEG Compression Loss, CQT Roundtrip Loss).
  - Displays the ASCII block diagram of processing pipelines.
  - Quick-start usage checklist.

---

### Page 2: Spectrogram Generation

- **Purpose**: Upload an audio file, inspect audio properties, adjust STFT parameters, and view generated 8-bit spectrogram images.
- **How to Operate**:
  1. Select audio source:
     - **Upload WAV File**: Drag and drop any `.wav` audio file.
     - **Generate Synthetic Test Signal**: Generates a harmonic A4/A5 sweep signal.
     - **Use Sample Audio**: Loads built-in pre-packaged sample audio (`data/input/sample.wav`).
  2. Adjust STFT parameters in the left sidebar:
     - **FFT Size (`n_fft`)**: `256`, `512`, `1024`, `2048`, `4096`
     - **Hop Length**: `64`, `128`, `256`, `512`, `1024`
     - **Window Function**: `hann`, `hamming`, `blackman`, `boxcar`
     - **Scaling Type**: `log1p` (\(\ln(1+x)\)) or `db` (\(20 \log_{10}(S)\))
  3. View output tabs:
     - **Linear Magnitude Spectrogram**: Displays raw floating-point STFT magnitude.
     - **Scaled Spectrogram**: Displays log/dB scaled magnitude matrix.
     - **8-bit Quantized PNG Image**: Displays uint8 `[0, 255]` image with JSON metadata (`scaleMin`, `scaleMax`, `fileMin`, `fileMax`).

---

### Page 3: Image Compression Experiment

- **Purpose**: Evaluates how JPEG compression degrades reconstructed audio quality compared to lossless PNG.
- **How to Operate**:
  1. Ensure an audio file is selected in Page 2.
  2. Select the reconstruction algorithm in the left sidebar (`Griffin-Lim`, `SPSI`, `Hybrid SPSI+GL`) and set Griffin-Lim iterations.
  3. Click **▶️ Run Image Compression Experiment**.
  4. Review the generated **Metric Comparison Table**:
     - **Format**: `PNG`, `JPEG 100`, `JPEG 80`, `JPEG 50`
     - **File Size (KB)**: Disk storage footprint
     - **SNR (dB)**: Signal-to-Noise Ratio
     - **PESQ**: Perceptual Evaluation of Speech Quality (0.5 to 4.5 MOS scale)
     - **STOI**: Short-Time Objective Intelligibility (0.0 to 1.0)
  5. Select a format from the **Visual & Acoustic Format Inspector** dropdown to view side-by-side spectrograms, error diff heatmaps, and play/download reconstructed audio.
  6. Click **📥 Export Results as CSV** to download experiment logs.

---

### Page 4: Constant-Q Transform (CQT) Experiment

- **Purpose**: Evaluates information loss introduced by non-linear `Linear -> CQT -> Linear` transformations.
- **How to Operate**:
  1. Adjust CQT parameters in the sidebar:
     - **Lowest FFT Row (`lowRow`)**: Lowest frequency bin to include (e.g., `10`).
     - **CQT Frequency Bins**: `128`, `256`, `513`, `1024`.
  2. Click **▶️ Run CQT Roundtrip Experiment**.
  3. Inspect metrics:
     - **Direct Linear SNR** vs **CQT Roundtrip SNR**.
     - **Mean Spectral Error**.
     - **PESQ** and **STOI** scores.
  4. Compare visual tabs:
     - **Original Linear Spectrogram**
     - **CQT Spectrogram** (logarithmic frequency axis)
     - **Reconstructed Linear Spectrogram** (derived from CQT)
     - **Difference Heatmap** (\(|Original - Reconstructed|\))
  5. Listen to audio playback comparing **Direct Linear Reconstruction** against **CQT Roundtrip Reconstruction**.

---

### Page 5: Audio Reconstruction

- **Purpose**: Compares phase retrieval algorithms and exports reconstructed audio files.
- **How to Operate**:
  1. Select a phase retrieval algorithm:
     - `Griffin-Lim`
     - `SPSI (Single-pass Single-sideband)`
     - `Hybrid SPSI + Griffin-Lim`
     - `Random Phase Baseline`
  2. Set Griffin-Lim iteration slider (`5` to `100`).
  3. Click **▶️ Synthesize Audio** to run the selected method and view its SNR, PESQ, and STOI metrics.
  4. Click **🔄 Benchmark All Phase Retrieval Algorithms** to run all 4 algorithms side-by-side and display a comparison table with audio players for each method.
  5. Click **📥 Download Reconstructed WAV File** to save the audio locally.

---

## 4. Using the Python Core API Programmatically

You can use the core signal-processing engine in your own Python scripts independently of Streamlit:

### Example: Audio Loading, STFT, Quantization, and Reconstruction

```python
from core.audio.loader import load_audio, save_audio
from core.spectrogram.generator import generate_spectrogram
from core.spectrogram.scaling import scaled_to_magnitude
from core.spectrogram.quantization import dequantize_from_8bit
from core.audio.reconstruction import reconstruct_audio
from core.metrics.snr import calculate_snr

# 1. Load Audio
y, sr, meta = load_audio("data/input/sample.wav")

# 2. Forward STFT & 8-bit Quantization
stft_complex, mag_sgram, scaled_sgram, uint8_img, q_meta = generate_spectrogram(
    y=y, n_fft=1024, hop_length=256, scale_type="log1p"
)

# 3. De-quantize 8-bit Image back to Magnitude
dequant_scaled = dequantize_from_8bit(uint8_img, q_meta["scaleMin"], q_meta["scaleMax"])
dequant_mag = scaled_to_magnitude(dequant_scaled, scale_type="log1p")

# 4. Phase Retrieval (Griffin-Lim)
y_recon = reconstruct_audio(
    msgram=dequant_mag, n_fft=1024, hop_length=256, method="griffin_lim", n_iter=50
)

# 5. Measure SNR
snr = calculate_snr(y, y_recon)
print(f"Reconstructed Audio SNR: {snr:.2f} dB")

# 6. Save Reconstructed Audio
save_audio("data/output/reconstructed.wav", y_recon, sr)
```

---

### Example: Linear to CQT and CQT to Linear Mapping

```python
from core.cqt.forward import linear_to_cqt
from core.cqt.inverse import cqt_to_linear

# 1. Convert Linear Spectrogram to CQT
cqt_sgram, cqt_meta = linear_to_cqt(
    linear_sgram=scaled_sgram, n_fft=1024, low_row=10, cqt_bins=513
)

# 2. Recover Linear Spectrogram from CQT
recon_linear_scaled = cqt_to_linear(
    cqt_sgram=cqt_sgram, target_bins=scaled_sgram.shape[0], low_row=10, metadata=cqt_meta
)
```

---

## 5. Running Unit Tests

To run all automated unit tests:

```powershell
$env:PYTHONPATH="."; python -m pytest tests/
```

### Verified Test Suites
- `tests/test_stft.py`: STFT and ISTFT roundtrip correctness.
- `tests/test_quantization.py`: 8-bit scaling and inverse scaling bounds.
- `tests/test_cqt.py`: Dan Ellis `logfmap` matrix construction and dimension validity.
- `tests/test_reconstruction.py`: Griffin-Lim, SPSI, Hybrid SPSI+GL, and Random Phase baseline algorithms.

---

## 6. Troubleshooting & FAQ

#### Q1: Why does PESQ show 0.0 for short audio files?
- **Answer**: PESQ algorithm requires audio signals of at least ~0.25 seconds in duration. For shorter audio, PESQ returns `0.0` automatically.

#### Q2: What is the difference between `log1p` and `db` scaling?
- **Answer**: `log1p` computes \(\ln(1 + |S|)\), which maps zero magnitude to zero. `db` computes \(20 \log_{10}(|S|)\) relative to maximum peak amplitude, which is standard in acoustic visualization.

#### Q3: Where are output images and audio files saved?
- **Answer**: Generated artifacts are automatically saved in `data/output/` and can be downloaded directly from the web interface.
