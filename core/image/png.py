import os
import json
import io
import numpy as np
from PIL import Image, PngImagePlugin
from typing import Tuple, Dict, Any, Union
from .preprocessing import array_to_image, image_to_array

def save_png_spectrogram(
    output_path: str,
    uint8_arr: np.ndarray,
    metadata: Dict[str, Any]
) -> Tuple[str, int]:
    """
    Save 8-bit uint8 spectrogram array to PNG image with embedded JSON metadata.
    
    Parameters:
        output_path: Path to target PNG file
        uint8_arr: 2D uint8 numpy array
        metadata: Metadata dictionary to embed in PNG text chunk
        
    Returns:
        (output_path, file_size_bytes)
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    img = array_to_image(uint8_arr, flip_vertical=True)
    
    info = PngImagePlugin.PngInfo()
    # Serialize metadata to JSON string
    info.add_text("meta", json.dumps(metadata))
    
    img.save(output_path, format="PNG", pnginfo=info)
    file_size = os.path.getsize(output_path)
    
    return output_path, file_size

def load_png_spectrogram(
    file_path: str
) -> Tuple[np.ndarray, Dict[str, Any], int]:
    """
    Load PNG spectrogram image and extract embedded JSON metadata.
    
    Parameters:
        file_path: Path to PNG file
        
    Returns:
        (uint8_arr, metadata, file_size_bytes)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PNG file not found: {file_path}")
        
    img = Image.open(file_path)
    file_size = os.path.getsize(file_path)
    
    metadata = {}
    if hasattr(img, "text") and "meta" in img.text:
        try:
            metadata = json.loads(img.text["meta"])
        except Exception:
            metadata = {}
            
    uint8_arr = image_to_array(img, flip_vertical=True)
    return uint8_arr, metadata, file_size

def load_and_validate_png(
    file_or_path: Union[str, io.BytesIO, Any]
) -> Tuple[np.ndarray, Dict[str, Any], int, Dict[str, Any]]:
    """
    Validate, load, convert color modes if needed, and extract embedded JSON metadata from PNG file or stream.
    
    Parameters:
        file_or_path: File path string or file-like bytes stream.
        
    Returns:
        uint8_arr: 2D uint8 numpy array [num_bins, num_frames]
        metadata: Embedded JSON metadata dictionary (or empty dict)
        file_size: Size in bytes
        image_info: Dictionary containing image dimensions and original color mode
    """
    try:
        if isinstance(file_or_path, str):
            if not os.path.exists(file_or_path):
                raise FileNotFoundError(f"File not found: {file_or_path}")
            file_size = os.path.getsize(file_or_path)
            img = Image.open(file_or_path)
        else:
            # File-like object (BytesIO or Streamlit UploadedFile)
            if hasattr(file_or_path, "size") and file_or_path.size is not None:
                file_size = file_or_path.size
            else:
                file_or_path.seek(0, os.SEEK_END)
                file_size = file_or_path.tell()
                file_or_path.seek(0)
            img = Image.open(file_or_path)
    except Exception as e:
        raise ValueError(f"Invalid image file: {str(e)}")

    original_mode = img.mode
    format_name = img.format if img.format else "PNG"
    
    # Extract embedded PNG text metadata
    metadata = {}
    if hasattr(img, "text") and "meta" in img.text:
        try:
            metadata = json.loads(img.text["meta"])
        except Exception:
            metadata = {}

    # Validate image dimensions
    width, height = img.size
    if width < 2 or height < 2:
        raise ValueError(f"Image dimensions too small ({width}x{height}). Minimum required size is 2x2 pixels.")

    # Convert RGB/RGBA/P color modes to grayscale 'L'
    if img.mode != "L":
        img_gray = img.convert("L")
    else:
        img_gray = img

    uint8_arr = image_to_array(img_gray, flip_vertical=True)

    image_info = {
        "format": format_name,
        "original_mode": original_mode,
        "width": width,
        "height": height,
        "num_bins": uint8_arr.shape[0],
        "num_frames": uint8_arr.shape[1],
        "has_embedded_metadata": len(metadata) > 0
    }

    return uint8_arr, metadata, file_size, image_info
