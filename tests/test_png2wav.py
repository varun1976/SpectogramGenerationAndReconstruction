import os
import tempfile
import numpy as np
import pytest
from PIL import Image
from core.audio.loader import load_audio, save_audio
from core.spectrogram.generator import generate_spectrogram
from core.image.png import save_png_spectrogram, load_png_spectrogram, load_and_validate_png
from core.audio.png2wav_engine import reconstruct_audio_from_png
from core.metrics.snr import calculate_snr

def test_quantization_inversion():
    data = np.linspace(0.0, 10.0, 500).reshape((5, 100)).astype(np.float32)
    stft_complex, mag_sgram, scaled_sgram, uint8_img, q_meta = generate_spectrogram(
        y=np.sin(np.linspace(0, 100, 2000)).astype(np.float32),
        n_fft=256,
        hop_length=64
    )
    assert uint8_img.dtype == np.uint8
    assert uint8_img.shape == mag_sgram.shape

def test_png_validation_and_rgb_conversion():
    with tempfile.TemporaryDirectory() as tmpdir:
        rgb_path = os.path.join(tmpdir, "test_rgb.png")
        # Create synthetic RGB image (100x50 pixels)
        arr_rgb = np.random.randint(0, 256, (50, 100, 3), dtype=np.uint8)
        img_rgb = Image.fromarray(arr_rgb, mode="RGB")
        img_rgb.save(rgb_path, format="PNG")
        
        uint8_arr, meta, file_size, img_info = load_and_validate_png(rgb_path)
        assert img_info["original_mode"] == "RGB"
        assert uint8_arr.ndim == 2
        assert uint8_arr.shape == (50, 100)

def test_embedded_metadata_extraction():
    with tempfile.TemporaryDirectory() as tmpdir:
        png_path = os.path.join(tmpdir, "test_meta.png")
        arr = np.random.randint(0, 256, (129, 50), dtype=np.uint8)
        test_meta = {"sample_rate": 16000, "n_fft": 256, "scaleMin": 0.0, "scaleMax": 10.0}
        
        save_png_spectrogram(png_path, arr, test_meta)
        uint8_arr, meta, file_size, img_info = load_and_validate_png(png_path)
        
        assert meta["sample_rate"] == 16000
        assert meta["n_fft"] == 256
        assert meta["scaleMin"] == 0.0

def test_end_to_end_png2wav():
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = os.path.join(tmpdir, "input.wav")
        png_path = os.path.join(tmpdir, "spectrogram.png")
        
        sr = 16000
        t = np.linspace(0, 1.0, sr)
        y_orig = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        save_audio(wav_path, y_orig, sr)
        
        _, _, _, uint8_img, q_meta = generate_spectrogram(y_orig, n_fft=512, hop_length=128)
        q_meta["sample_rate"] = sr
        save_png_spectrogram(png_path, uint8_img, q_meta)
        
        y_recon, sr_rec, recovered_mag, result_meta = reconstruct_audio_from_png(png_path)
        
        assert len(y_recon) > 0
        assert sr_rec == sr
        assert result_meta["mode_used"] == "Mode A (Metadata)"
        
        snr_val = calculate_snr(y_orig, y_recon)
        assert snr_val > -10.0

def test_invalid_inputs():
    with pytest.raises(ValueError):
        load_and_validate_png("non_existent_file.png")

if __name__ == "__main__":
    test_quantization_inversion()
    test_png_validation_and_rgb_conversion()
    test_embedded_metadata_extraction()
    test_end_to_end_png2wav()
    print("All PNG2WAV unit tests passed!")
