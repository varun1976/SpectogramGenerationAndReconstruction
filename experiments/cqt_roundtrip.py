import time
import numpy as np
from typing import Dict, Any, Tuple
from core.spectrogram.generator import generate_spectrogram
from core.spectrogram.scaling import scaled_to_magnitude
from core.spectrogram.quantization import dequantize_from_8bit, quantize_to_8bit
from core.cqt.forward import linear_to_cqt
from core.cqt.inverse import cqt_to_linear
from core.audio.reconstruction import reconstruct_audio
from core.metrics.snr import calculate_snr
from core.metrics.stoi import calculate_stoi
from core.metrics.pesq import calculate_pesq

def run_cqt_roundtrip_experiment(
    y: np.ndarray,
    sr: int,
    n_fft: int = 1024,
    hop_length: int = 256,
    scale_type: str = "log1p",
    low_row: int = 10,
    cqt_bins: int = 513,
    recon_method: str = "griffin_lim",
    n_iter: int = 50
) -> Dict[str, Any]:
    """
    Evaluate information loss in Linear -> CQT -> Linear roundtrip transformation.
    """
    t0 = time.time()
    
    # 1. Linear Spectrogram Generation
    stft_complex, mag_linear, scaled_linear, uint8_linear, q_meta = generate_spectrogram(
        y=y,
        n_fft=n_fft,
        hop_length=hop_length,
        scale_type=scale_type
    )
    
    # Direct Linear audio reconstruction baseline
    y_direct_recon = reconstruct_audio(
        msgram=mag_linear,
        n_fft=n_fft,
        hop_length=hop_length,
        method=recon_method,
        n_iter=n_iter,
        target_length=len(y)
    )
    
    # 2. Linear -> CQT Transformation
    cqt_scaled, cqt_meta = linear_to_cqt(
        linear_sgram=scaled_linear,
        n_fft=n_fft,
        low_row=low_row,
        cqt_bins=cqt_bins
    )
    uint8_cqt, q_cqt_meta = quantize_to_8bit(cqt_scaled)
    
    # 3. CQT Image Dequantization & CQT -> Linear Inverse Mapping
    dequant_cqt_scaled = dequantize_from_8bit(
        uint8_cqt,
        scale_min=q_cqt_meta["scaleMin"],
        scale_max=q_cqt_meta["scaleMax"]
    )
    
    recon_scaled_linear = cqt_to_linear(
        cqt_sgram=dequant_cqt_scaled,
        target_bins=mag_linear.shape[0],
        low_row=low_row,
        metadata=cqt_meta
    )
    recon_mag_linear = scaled_to_magnitude(recon_scaled_linear, scale_type=scale_type)
    
    # 4. Reconstruct audio from CQT roundtrip linear spectrogram
    y_cqt_recon = reconstruct_audio(
        msgram=recon_mag_linear,
        n_fft=n_fft,
        hop_length=hop_length,
        method=recon_method,
        n_iter=n_iter,
        target_length=len(y)
    )
    
    # Metrics
    snr_direct = calculate_snr(y, y_direct_recon)
    snr_cqt = calculate_snr(y, y_cqt_recon)
    snr_cqt_vs_direct = calculate_snr(y_direct_recon, y_cqt_recon)
    
    try:
        stoi_direct = calculate_stoi(y, y_direct_recon, sr)
    except Exception:
        stoi_direct = 0.0
        
    try:
        stoi_cqt = calculate_stoi(y, y_cqt_recon, sr)
    except Exception:
        stoi_cqt = 0.0
        
    try:
        pesq_direct = calculate_pesq(y, y_direct_recon, sr)
    except Exception:
        pesq_direct = 0.0
        
    try:
        pesq_cqt = calculate_pesq(y, y_cqt_recon, sr)
    except Exception:
        pesq_cqt = 0.0
    
    diff_sgram = np.abs(mag_linear - recon_mag_linear)
    
    elapsed = time.time() - t0
    
    return {
        "original_mag_linear": mag_linear,
        "cqt_scaled": cqt_scaled,
        "uint8_cqt": uint8_cqt,
        "recon_mag_linear": recon_mag_linear,
        "diff_sgram": diff_sgram,
        "y_direct_recon": y_direct_recon,
        "y_cqt_recon": y_cqt_recon,
        "metrics": {
            "snr_direct_db": snr_direct,
            "snr_cqt_db": snr_cqt,
            "snr_cqt_vs_direct_db": snr_cqt_vs_direct,
            "stoi_direct": stoi_direct,
            "stoi_cqt": stoi_cqt,
            "pesq_direct": pesq_direct,
            "pesq_cqt": pesq_cqt,
            "mean_diff": float(np.mean(diff_sgram)),
            "max_diff": float(np.max(diff_sgram)),
            "processing_time_sec": elapsed
        }
    }
