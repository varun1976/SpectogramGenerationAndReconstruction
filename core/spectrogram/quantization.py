import numpy as np
import math
from typing import Tuple, Dict, Optional, Any

def get_quantization_bounds(
    scaled_sgram: np.ndarray,
    round_to_int: bool = True
) -> Tuple[float, float, float, float]:
    """
    Find dataset or image min/max floating point values and integer scale bounds.
    
    Parameters:
        scaled_sgram: Scaled magnitude matrix
        round_to_int: Whether to floor min and ceil max for clean integer bounds
        
    Returns:
        (min_val, max_val, scale_min, scale_max)
    """
    min_val = float(np.min(scaled_sgram))
    max_val = float(np.max(scaled_sgram))
    
    if round_to_int:
        scale_min = float(math.floor(min_val))
        scale_max = float(math.ceil(max_val))
        # Ensure non-zero range
        if scale_max <= scale_min:
            scale_max = scale_min + 1.0
    else:
        scale_min = min_val
        scale_max = max_val if max_val > min_val else min_val + 1e-5
        
    return min_val, max_val, scale_min, scale_max

def quantize_to_8bit(
    scaled_sgram: np.ndarray,
    scale_min: Optional[float] = None,
    scale_max: Optional[float] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Quantize floating point scaled spectrogram to 8-bit uint8 array in range [0, 255].
    
    Parameters:
        scaled_sgram: Floating point scaled spectrogram matrix
        scale_min: Lower bound for scaling (if None, derived from data)
        scale_max: Upper bound for scaling (if None, derived from data)
        
    Returns:
        uint8_array: 2D uint8 numpy array [0, 255]
        meta: Metadata dictionary containing scaling parameters
    """
    raw_min, raw_max, calc_smin, calc_smax = get_quantization_bounds(scaled_sgram)
    
    s_min = float(scale_min) if scale_min is not None else calc_smin
    s_max = float(scale_max) if scale_max is not None else calc_smax
    
    range_val = s_max - s_min
    if range_val <= 0:
        range_val = 1.0
        
    norm_sgram = (scaled_sgram - s_min) / range_val
    uint8_img = np.round(255.0 * np.clip(norm_sgram, 0.0, 1.0)).astype(np.uint8)
    
    metadata = {
        "fileMin": raw_min,
        "fileMax": raw_max,
        "scaleMin": s_min,
        "scaleMax": s_max,
        "quantization": "8-bit [0-255]"
    }
    return uint8_img, metadata

def dequantize_from_8bit(
    uint8_img: np.ndarray,
    scale_min: float,
    scale_max: float
) -> np.ndarray:
    """
    Convert 8-bit uint8 spectrogram image array back to original floating point scale.
    
    Parameters:
        uint8_img: 2D uint8 image array
        scale_min: Original min scale factor
        scale_max: Original max scale factor
        
    Returns:
        De-quantized floating point spectrogram matrix
    """
    range_val = float(scale_max) - float(scale_min)
    img_float = uint8_img.astype(np.float32)
    dequant = (img_float / 255.0) * range_val + float(scale_min)
    return dequant.astype(np.float32)
