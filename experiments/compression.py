import os
import time
import tempfile
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from core.spectrogram.generator import generate_spectrogram
from core.spectrogram.scaling import scaled_to_magnitude
from core.spectrogram.quantization import dequantize_from_8bit
from core.image.png import save_png_spectrogram, load_png_spectrogram
from core.image.jpeg import save_jpeg_spectrogram, load_jpeg_spectrogram
from core.audio.reconstruction import reconstruct_audio
from core.metrics.snr import calculate_snr
from core.metrics.stoi import calculate_stoi
from core.metrics.pesq import calculate_pesq

def run_compression_experiment(
    y: np.ndarray,
    sr: int,
    n_fft: int = 1024,
    hop_length: int = 256,
    scale_type: str = "log1p",
    recon_method: str = "griffin_lim",
    n_iter: int = 50,
    temp_dir: str = "./data/output/compression"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Evaluate PNG vs JPEG (100, 80, 50) compression artifacts on audio reconstruction.
    """
    os.makedirs(temp_dir, exist_ok=True)
    
    stft_complex, mag_sgram, scaled_sgram, uint8_img, q_meta = generate_spectrogram(
        y=y,
        n_fft=n_fft,
        hop_length=hop_length,
        scale_type=scale_type
    )
    
    formats = [
        ("PNG", "png", None),
        ("JPEG 100", "jpg", 100),
        ("JPEG 80", "jpg", 80),
        ("JPEG 50", "jpg", 50)
    ]
    
    results_list = []
    artifacts = {}
    
    for fmt_label, ext, quality in formats:
        t0 = time.time()
        file_path = os.path.join(temp_dir, f"spectrogram_{fmt_label.lower().replace(' ', '_')}.{ext}")
        
        if ext == "png":
            save_png_spectrogram(file_path, uint8_img, q_meta)
            loaded_img, loaded_meta, file_size = load_png_spectrogram(file_path)
        else:
            save_jpeg_spectrogram(file_path, uint8_img, q_meta, quality=quality)
            loaded_img, loaded_meta, file_size = load_jpeg_spectrogram(file_path)
            
        s_min = loaded_meta.get("scaleMin", q_meta["scaleMin"])
        s_max = loaded_meta.get("scaleMax", q_meta["scaleMax"])
        
        dequant_scaled = dequantize_from_8bit(loaded_img, s_min, s_max)
        dequant_mag = scaled_to_magnitude(dequant_scaled, scale_type=scale_type)
        
        y_recon = reconstruct_audio(
            msgram=dequant_mag,
            n_fft=n_fft,
            hop_length=hop_length,
            method=recon_method,
            n_iter=n_iter,
            target_length=len(y)
        )
        
        try:
            snr_val = calculate_snr(y, y_recon)
        except Exception:
            snr_val = 0.0
            
        try:
            stoi_val = calculate_stoi(y, y_recon, sr)
        except Exception:
            stoi_val = 0.0
            
        try:
            pesq_val = calculate_pesq(y, y_recon, sr)
        except Exception:
            pesq_val = 0.0
        
        proc_time = time.time() - t0
        
        results_list.append({
            "Format": fmt_label,
            "File Size (KB)": round(file_size / 1024.0, 2),
            "SNR (dB)": round(snr_val, 2),
            "PESQ": round(pesq_val, 2),
            "STOI": round(stoi_val, 4),
            "Processing Time (s)": round(proc_time, 3)
        })
        
        artifacts[fmt_label] = {
            "file_path": file_path,
            "file_size_bytes": file_size,
            "img_array": loaded_img,
            "dequant_mag": dequant_mag,
            "reconstructed_audio": y_recon
        }
        
    df_results = pd.DataFrame(results_list)
    
    experiment_data = {
        "original_mag": mag_sgram,
        "uint8_img": uint8_img,
        "artifacts": artifacts
    }
    
    return df_results, experiment_data
