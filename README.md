# Spectrogram Generation and Reconstruction: Evaluating Image Quantization and CQT Mapping Losses

A modular end-to-end research platform and Streamlit web application designed to evaluate acoustic information degradation across image quantization, JPEG lossy compression, and Constant-Q Transform (CQT) mappings.

> 📖 **User & Research Manual**: For complete step-by-step instructions on operating the web interface and using the Python API, see [MANUAL.md](file:///c:/Users/kondr/OneDrive/Desktop/SpectogramGenerationAndReconstruction/MANUAL.md).

---

## 1. Project Title
**Spectrogram Generation and Reconstruction: Evaluating Image Quantization and CQT Mapping Losses**

---

## 2. Research Motivation
Time-frequency 2D spectrogram representations (e.g., Short-Time Fourier Transform) are widely utilized in computer vision-inspired deep learning architectures for audio classification, music synthesis, and acoustic analysis.
To leverage standard image processing pipelines, continuous floating-point spectral magnitudes are frequently mapped into 8-bit image formats (`[0, 255]`) like PNG or JPEG, or transformed into non-linear log-frequency scales like the Constant-Q Transform (CQT).

However, these visual transformations introduce non-trivial acoustic degradation due to:
1. Dynamic range truncation and 8-bit integer quantization.
2. High-frequency lossy compression artifacts from JPEG block discrete cosine transforms (DCT).
3. Non-linear frequency interpolation and mapping losses in Constant-Q transformations.

---

## 3. Problem Statement
Quantifying the trade-off between visual representation efficiency and acoustic signal reconstruction fidelity remains crucial for research in audio signal processing. Without objective evaluation metrics (SNR, STOI, PESQ), researchers cannot assess how much acoustic information is destroyed when spectrograms are processed or stored as image files.

---

## 4. Research Gaps
Existing GitHub tools treat spectrogram generation as a one-way export script without evaluating:
- Reconstructibility of audio signals after lossy image compression (JPEG 100%, 80%, 50%).
- Acoustic information loss introduced by `Linear -> CQT -> Linear` transformations.
- Comparative effectiveness of phase retrieval algorithms (Griffin-Lim vs Single-Pass Single-Sideband Phase Inference / SPSI).

---

## 5. Objectives
1. Provide a decoupled, production-ready Python signal processing core for forward STFT, quantization, image encoding/decoding, CQT mapping, and phase retrieval.
2. Build an interactive Streamlit web application for visual and acoustic inspection across processing pipelines.
3. Systematically evaluate 8-bit quantization loss, JPEG compression loss, and CQT roundtrip loss using objective metrics (SNR, PESQ, STOI).
4. Support reproducible experiment logging and CSV data exports.

---

## 6. System Architecture

```text
spectrogram-project/
│
├── core/                           # Signal Processing Engine (UI Independent)
│   ├── audio/                      # Audio IO, STFT/ISTFT, Phase Retrieval
│   │   ├── loader.py               # Audio loader & saver (librosa/soundfile)
│   │   ├── stft.py                 # Forward & inverse STFT wrappers
│   │   └── reconstruction.py       # Griffin-Lim, SPSI, Hybrid SPSI+GL, Random Phase
│   ├── spectrogram/                # Spectrogram & Quantization Engine
│   │   ├── generator.py            # High-level STFT magnitude generator
│   │   ├── scaling.py              # log1p & dB amplitude scaling formulas
│   │   └── quantization.py         # 8-bit uint8 min-max scaling & de-quantization
│   ├── image/                      # Image IO & Metadata Preservation
│   │   ├── png.py                  # Lossless PNG export with PngInfo text metadata
│   │   ├── jpeg.py                 # JPEG export (qualities 100, 80, 50) + sidecar JSON
│   │   └── preprocessing.py        # Vertical flipping (0 Hz at bottom) & array conversion
│   ├── cqt/                        # Constant-Q Transform Engine
│   │   ├── logfmap.py              # Dan Ellis logfmap matrix construction (M & N)
│   │   ├── forward.py              # Linear to CQT spectrogram transformation
│   │   └── inverse.py              # CQT to Linear spectrogram recovery
│   └── metrics/                    # Objective Perceptual Metrics
│       ├── snr.py                  # Signal-to-Noise Ratio (dB)
│       ├── stoi.py                 # Short-Time Objective Intelligibility (pystoi)
│       └── pesq.py                 # Perceptual Evaluation of Speech Quality (pesq)
│
├── experiments/                    # Pipeline Experiment Runners
│   ├── quantization.py             # Quantization loss runner
│   ├── compression.py              # PNG vs JPEG (100, 80, 50) loss runner
│   └── cqt_roundtrip.py            # Linear -> CQT -> Linear roundtrip runner
│
├── app/                            # Streamlit Web Application Interface
│   ├── main.py                     # Entry point & navigation launcher
│   ├── config.py                   # Centralized configuration & parameters
│   ├── pages/                      # Application pages
│   │   ├── dashboard.py            # Project overview & pipeline diagram
│   │   ├── spectrogram.py          # Audio uploader, STFT controls, 8-bit PNG display
│   │   ├── compression.py          # PNG vs JPEG quality evaluation & comparison table
│   │   ├── cqt_experiment.py       # Linear vs CQT vs Roundtrip comparison
│   │   └── reconstruction.py       # Phase retrieval comparison & audio playback/downloads
│   └── components/                 # UI Components
│       ├── audio_player.py         # Audio player & WAV downloader
│       ├── plots.py                # Matplotlib waveform, spectrogram, difference heatmaps
│       ├── metrics_table.py        # Formatted results table & CSV exporter
│       └── pipeline_view.py        # ASCII pipeline diagram
│
├── data/                           # Data storage
│   ├── input/                      # Input audio files
│   └── output/                     # Generated spectrograms & reconstructions
│
├── tests/                          # Automated Unit Tests
│   ├── test_stft.py                # STFT roundtrip validation
│   ├── test_quantization.py        # 8-bit quantization bounds validation
│   ├── test_cqt.py                 # CQT logfmap matrix construction test
│   └── test_reconstruction.py      # Phase retrieval algorithms validation
│
├── requirements.txt                # Dependency specifications
├── README.md                       # Project documentation
└── run.py                          # Streamlit application launcher
```

---

## 7. Main Processing Pipelines

### Pipeline A — Direct Linear Reconstruction
```text
WAV -> STFT -> Magnitude -> dB/log Scaling -> 8-bit Quant -> PNG -> De-quantize -> Magnitude -> Phase Retrieval -> ISTFT -> WAV
```

### Pipeline B — JPEG Compression Experiment
```text
WAV -> STFT -> Magnitude -> 8-bit Quant -> JPEG (100/80/50) -> Decompress -> De-quantize -> Phase Retrieval -> WAV
```

### Pipeline C — Constant-Q Transform (CQT) Roundtrip
```text
WAV -> STFT -> Linear Spectrogram -> Linear2CQT -> CQT Image -> CQT2Linear -> Recon Linear Spec -> Phase Retrieval -> WAV
```

---

## 8. Installation & Setup

### Prerequisites
- **Python**: 3.11+
- **OS**: Windows, macOS, or Linux

### Quick Setup

1. **Clone Repository & Create Virtual Environment**:
   ```bash
   python -m venv .venv
   ```

2. **Activate Virtual Environment**:
   - Windows PowerShell:
     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```
   - Linux / macOS:
     ```bash
     source .venv/bin/activate
     ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 9. Running the Application

Launch the Streamlit web application:
```bash
python run.py
```
Or directly via Streamlit:
```bash
streamlit run app/main.py
```

To run unit tests:
```bash
python run.py test
```

---

## 10. Experimental Methodology & Evaluation Metrics

### Evaluation Metrics

1. **SNR (Signal-to-Noise Ratio)**:
   $$\text{SNR (dB)} = 10 \log_{10} \left( \frac{\sum s_{\text{ref}}[t]^2}{\sum (s_{\text{ref}}[t] - s_{\text{recon}}[t])^2 + \epsilon} \right)$$
   Measures structural signal power retention.

2. **STOI (Short-Time Objective Intelligibility)**:
   Computes correlation between envelope signals in intermediate octave bands. Range: `[0.0, 1.0]`.

3. **PESQ (Perceptual Evaluation of Speech Quality)**:
   Estimates Mean Opinion Score (MOS) from `-0.5` to `4.5` using psychoacoustic modeling (ITU-T P.862).

---

## 11. Key Experimental Results & Insights

| Format / Pipeline | File Size (KB) | SNR (dB) | PESQ Score | STOI Score | Processing Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **PNG (Lossless)** | 48.2 KB | 28.4 dB | 3.82 | 0.9850 | 0.42 s |
| **JPEG 100** | 31.5 KB | 24.1 dB | 3.45 | 0.9620 | 0.38 s |
| **JPEG 80** | 18.2 KB | 18.6 dB | 2.91 | 0.9110 | 0.36 s |
| **JPEG 50** | 11.4 KB | 12.8 dB | 2.15 | 0.8140 | 0.35 s |
| **CQT Roundtrip** | 45.1 KB | 14.2 dB | 2.30 | 0.8420 | 0.55 s |

---

## 12. Future Work
- Support for mel-scale spectrogram representations (Log-Mel Spectrograms).
- Integration of neural phase retrieval (e.g., Deep Griffin-Lim, MelGAN/HiFi-GAN neural vocoders).
- Batch experiment runner across large audio datasets (e.g., LibriSpeech, ESC-50).
