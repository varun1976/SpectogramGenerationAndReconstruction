import numpy as np
from core.spectrogram.quantization import quantize_to_8bit, dequantize_from_8bit

def test_quantization_dequantization():
    # Synthetic scaled spectrogram matrix
    data = np.linspace(-10.0, 50.0, 1000).reshape((10, 100)).astype(np.float32)
    
    uint8_img, meta = quantize_to_8bit(data)
    assert uint8_img.dtype == np.uint8
    assert np.min(uint8_img) >= 0
    assert np.max(uint8_img) <= 255
    
    dequant = dequantize_from_8bit(uint8_img, meta["scaleMin"], meta["scaleMax"])
    
    # 8-bit quantization step size is (scaleMax - scaleMin) / 255
    step_size = (meta["scaleMax"] - meta["scaleMin"]) / 255.0
    max_error = np.max(np.abs(data - dequant))
    
    assert max_error <= step_size + 1e-4

if __name__ == "__main__":
    test_quantization_dequantization()
    print("test_quantization_dequantization passed!")
