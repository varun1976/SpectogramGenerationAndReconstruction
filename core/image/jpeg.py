import os
import json
import numpy as np
from PIL import Image
from typing import Tuple, Dict, Any
from .preprocessing import array_to_image, image_to_array

def save_jpeg_spectrogram(
    output_path: str,
    uint8_arr: np.ndarray,
    metadata: Dict[str, Any],
    quality: int = 100
) -> Tuple[str, int]:
    """
    Save 8-bit uint8 spectrogram array to JPEG image with specified quality (1-100) and sidecar JSON metadata.
    
    Parameters:
        output_path: Path to target JPEG file (.jpg or .jpeg)
        uint8_arr: 2D uint8 numpy array
        metadata: Metadata dictionary
        quality: JPEG compression quality (50, 80, 100, etc.)
        
    Returns:
        (output_path, file_size_bytes)
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    img = array_to_image(uint8_arr, flip_vertical=True)
    img.save(output_path, format="JPEG", quality=quality)
    file_size = os.path.getsize(output_path)
    
    # Save sidecar metadata JSON file
    meta_path = output_path + ".meta.json"
    jpeg_metadata = dict(metadata)
    jpeg_metadata["jpeg_quality"] = quality
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(jpeg_metadata, f, indent=2)
        
    return output_path, file_size

def load_jpeg_spectrogram(
    file_path: str
) -> Tuple[np.ndarray, Dict[str, Any], int]:
    """
    Load JPEG spectrogram image and sidecar JSON metadata.
    
    Parameters:
        file_path: Path to JPEG file
        
    Returns:
        (uint8_arr, metadata, file_size_bytes)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"JPEG file not found: {file_path}")
        
    img = Image.open(file_path)
    file_size = os.path.getsize(file_path)
    
    metadata = {}
    meta_path = file_path + ".meta.json"
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            metadata = {}
            
    uint8_arr = image_to_array(img, flip_vertical=True)
    return uint8_arr, metadata, file_size
