import json
import os
import io
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union
from core.image.png import load_and_validate_png
from core.spectrogram.quantization import dequantize_from_8bit
from core.spectrogram.scaling import scaled_to_magnitude
from core.audio.reconstruction import reconstruct_audio

def reconstruct_audio_from_png(
    image_input: Union[str, io.BytesIO, Any],
    metadata_input: Optional[Union[str, io.BytesIO, Dict[str, Any]]] = None,
    manual_params: Optional[Dict[str, Any]] = None
) -> Tuple[np.ndarray, int, np.ndarray, Dict[str, Any]]:
    """
    Reconstruct audio signal from a PNG spectrogram image.
    Supports Mode A (Automatic Metadata) and Mode B (Manual Parameters).
    
    Parameters:
        image_input: File path string or file-like bytes stream of PNG image.
        metadata_input: Optional JSON file path, bytes stream, or dictionary containing STFT/scaling metadata.
        manual_params: Optional dictionary of manual parameter overrides.
        
    Returns:
        y_recon: Reconstructed audio signal array (float32)
        sr: Sample rate
        recovered_mag: De-quantized linear magnitude spectrogram matrix [num_bins, num_frames]
        result_meta: Combined metadata dictionary describing parameters used and mode
    """
    # 1. Validate and load PNG image array & embedded metadata
    uint8_arr, embedded_meta, file_size, image_info = load_and_validate_png(image_input)
    
    # 2. Parse external metadata file/dict if provided
    external_meta = {}
    if metadata_input is not None:
        if isinstance(metadata_input, dict):
            external_meta = dict(metadata_input)
        elif isinstance(metadata_input, str) and os.path.exists(metadata_input):
            try:
                with open(metadata_input, "r", encoding="utf-8") as f:
                    external_meta = json.load(f)
            except Exception as e:
                raise ValueError(f"Could not parse sidecar metadata JSON file: {str(e)}")
        elif hasattr(metadata_input, "read"):
            try:
                content = metadata_input.read()
                if isinstance(content, bytes):
                    content = content.decode("utf-8")
                external_meta = json.loads(content)
            except Exception as e:
                raise ValueError(f"Could not parse uploaded metadata JSON: {str(e)}")

    # Combine embedded metadata and external metadata (external takes precedence)
    combo_meta = {**embedded_meta, **external_meta}
    
    manual_params = manual_params or {}
    
    # Determine operation mode
    if len(combo_meta) > 0 and not manual_params.get("force_manual", False):
        mode_used = "Mode A (Metadata)"
    else:
        mode_used = "Mode B (Manual Parameters)"

    num_bins, num_frames = uint8_arr.shape
    
    # Infer n_fft from image height if unspecified: height = n_fft // 2 + 1 => n_fft = 2 * (height - 1)
    inferred_n_fft = 2 * (num_bins - 1)
    
    # Resolve parameters with cascading priority: manual_params > combo_meta > defaults
    sr = int(manual_params.get("sample_rate", combo_meta.get("sample_rate", 22050)))
    n_fft = int(manual_params.get("n_fft", combo_meta.get("n_fft", inferred_n_fft)))
    hop_length = int(manual_params.get("hop_length", combo_meta.get("hop_length", n_fft // 4)))
    win_length = int(manual_params.get("win_length", combo_meta.get("win_length", n_fft)))
    window = str(manual_params.get("window", combo_meta.get("window", "hann")))
    scale_type = str(manual_params.get("scale_type", combo_meta.get("scale_type", "log1p")))
    
    scale_min = float(manual_params.get("scale_min", combo_meta.get("scaleMin", combo_meta.get("scale_min", 0.0))))
    scale_max = float(manual_params.get("scale_max", combo_meta.get("scaleMax", combo_meta.get("scale_max", 255.0))))
    
    recon_method = str(manual_params.get("recon_method", combo_meta.get("recon_method", "griffin_lim")))
    n_iter = int(manual_params.get("n_iter", combo_meta.get("gl_iterations", combo_meta.get("n_iter", 50))))
    
    # 3. Inverse 8-bit De-quantization
    dequant_scaled = dequantize_from_8bit(uint8_arr, scale_min=scale_min, scale_max=scale_max)
    
    # 4. Inverse Magnitude Scaling (log1p or dB -> Linear Magnitude)
    recovered_mag = scaled_to_magnitude(dequant_scaled, scale_type=scale_type)
    
    # Target audio output length (approximate)
    target_length = num_frames * hop_length
    
    # 5. Phase Retrieval and Audio Synthesis
    y_recon = reconstruct_audio(
        msgram=recovered_mag,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        method=recon_method,
        n_iter=n_iter,
        target_length=target_length
    )
    
    result_meta = {
        "mode_used": mode_used,
        "sample_rate": sr,
        "n_fft": n_fft,
        "hop_length": hop_length,
        "win_length": win_length,
        "window": window,
        "scale_type": scale_type,
        "scale_min": scale_min,
        "scale_max": scale_max,
        "recon_method": recon_method,
        "n_iter": n_iter,
        "file_size": file_size,
        "image_info": image_info,
        "duration_sec": len(y_recon) / sr
    }
    
    return y_recon, sr, recovered_mag, result_meta
