import os
import time
import numpy as np
from typing import Dict, Any, Tuple
from core.audio.stft import compute_stft
from core.spectrogram.generator import generate_spectrogram
from core.spectrogram.scaling import scaled_to_magnitude
from core.spectrogram.quantization import dequantize_from_8bit
from core.audio.reconstruction import reconstruct_audio
from core.metrics.snr import calculate_snr
from core.metrics.stoi import calculate_stoi
from core.metrics.pesq import calculate_pesq

def run_quantization_experiment(
    y: np.ndarray,
    sr: int,
    n_fft: int = 1024,
    hop_length: int = 256,
    scale_type: str = "log1p",
    recon_method: str = "griffin_lim",
    n_iter: int = 50
) -> Dict[str, Any]:
    """
    Evaluate information loss introduced by 8-bit quantization [0, 255].
    
    Compares:
      1. Float magnitude reconstruction (unquantized baseline)
      2. 8-bit quantized image reconstruction
    """
    start_time = time.time()
    
    # 1. Forward STFT & spectrogram generation
    stft_complex, mag_sgram, scaled_sgram, uint8_img, q_meta = generate_spectrogram(
        y=y,
        n_fft=n_fft,
        hop_length=hop_length,
        scale_type=scale_type
    )
    
    # Baseline reconstruction directly from continuous magnitude
    y_float_recon = reconstruct_audio(
        msgram=mag_sgram,
        n_fft=n_fft,
        hop_length=hop_length,
        method=recon_method,
        n_iter=n_iter,
        target_length=len(y)
    )
    
    # 2. De-quantize uint8 image back to scaled float
    dequant_scaled = dequantize_from_8bit(
        uint8_img=uint8_img,
        scale_min=q_meta["scaleMin"],
        scale_max=q_meta["scaleMax"]
    )
    dequant_mag = scaled_to_magnitude(dequant_scaled, scale_type=scale_type)
    
    # Reconstruct audio from dequantized magnitude
    y_quant_recon = reconstruct_audio(
        msgram=dequant_mag,
        n_fft=n_fft,
        hop_length=hop_length,
        method=recon_method,
        n_iter=n_iter,
        target_length=len(y)
    )
    
    # Metrics
    float_snr = calculate_snr(y, y_float_recon)
    quant_snr = calculate_snr(y, y_quant_recon)
    quant_vs_float_snr = calculate_snr(y_float_recon, y_quant_recon)
    
    try:
        float_stoi = calculate_stoi(y, y_float_recon, sr)
    except Exception:
        float_stoi = 0.0
        
    try:
        quant_stoi = calculate_stoi(y, y_quant_recon, sr)
    except Exception:
        quant_stoi = 0.0
        
    try:
        float_pesq = calculate_pesq(y, y_float_recon, sr)
    except Exception:
        float_pesq = 0.0
        
    try:
        quant_pesq = calculate_pesq(y, y_quant_recon, sr)
    except Exception:
        quant_pesq = 0.0
    
    mag_diff = np.abs(mag_sgram - dequant_mag)
    mean_quant_error = float(np.mean(mag_diff))
    max_quant_error = float(np.max(mag_diff))
    
    elapsed = time.time() - start_time
    
    return {
        "float_reconstruction": y_float_recon,
        "quant_reconstruction": y_quant_recon,
        "uint8_img": uint8_img,
        "mag_sgram": mag_sgram,
        "dequant_mag": dequant_mag,
        "mag_diff": mag_diff,
        "metrics": {
            "float_snr_db": float_snr,
            "quant_snr_db": quant_snr,
            "quant_vs_float_snr_db": quant_vs_float_snr,
            "float_stoi": float_stoi,
            "quant_stoi": quant_stoi,
            "float_pesq": float_pesq,
            "quant_pesq": quant_pesq,
            "mean_mag_error": mean_quant_error,
            "max_mag_error": max_quant_error,
            "processing_time_sec": elapsed
        }
    }
