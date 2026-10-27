import numpy as np
from PIL import Image

def array_to_image(uint8_arr: np.ndarray, flip_vertical: bool = True) -> Image.Image:
    """
    Convert 2D uint8 numpy array [num_bins, num_frames] into a PIL Image ('L' mode).
    
    Parameters:
        uint8_arr: 2D uint8 array
        flip_vertical: If True, flips array vertically so 0 Hz is at the bottom (standard spectrogram display)
        
    Returns:
        PIL Image instance in grayscale ('L') mode
    """
    if flip_vertical:
        arr_display = np.flipud(uint8_arr)
    else:
        arr_display = uint8_arr
        
    img = Image.fromarray(arr_display).convert('L')
    return img

def image_to_array(img: Image.Image, flip_vertical: bool = True) -> np.ndarray:
    """
    Convert PIL Image ('L' mode) into a 2D uint8 numpy array [num_bins, num_frames].
    
    Parameters:
        img: PIL Image instance
        flip_vertical: If True, flips array vertically to restore original frequency axis orientiation
        
    Returns:
        2D uint8 numpy array
    """
    img_gray = img.convert('L')
    arr = np.asarray(img_gray, dtype=np.uint8)
    
    if flip_vertical:
        arr = np.flipud(arr)
        
    return arr.copy()
