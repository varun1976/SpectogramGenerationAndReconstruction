import numpy as np
import librosa
import sys

def calculate_pesq(
    ref_signal: np.ndarray,
    deg_signal: np.ndarray,
    sr: int,
    mode: str = "wb"
) -> float:
    """
    Calculate Perceptual Evaluation of Speech Quality (PESQ) score.
    
    1. Tries C-extension `pesq` library if available in environment.
    2. Uses native NumPy/SciPy `pesq_pure_python` implementation if C-extension is missing.
    
    Parameters:
        ref_signal: Reference original audio array
        deg_signal: Degraded/reconstructed audio array
        sr: Sample rate in Hz
        mode: 'wb' (wideband, 16 kHz) or 'nb' (narrowband, 8 kHz)
        
    Returns:
        PESQ MOS score in range [1.0, 4.5]
    """
    # 1. Try PyPI pesq C-extension if installed
    pesq_c = None
    try:
        import pesq as pesq_c
    except Exception as e:
        pesq_c = None
        
    if pesq_c is not None:
        try:
            ref = ref_signal.astype(np.float32).flatten()
            deg = deg_signal.astype(np.float32).flatten()
            
            min_len = min(len(ref), len(deg))
            if min_len >= int(0.25 * sr):
                ref = ref[:min_len]
                deg = deg[:min_len]
                
                target_sr = 16000 if mode == "wb" else 8000
                if sr != target_sr:
                    ref_resample = librosa.resample(ref, orig_sr=sr, target_sr=target_sr)
                    deg_resample = librosa.resample(deg, orig_sr=sr, target_sr=target_sr)
                else:
                    ref_resample = ref
                    deg_resample = deg
                    
                min_r_len = min(len(ref_resample), len(deg_resample))
                ref_resample = ref_resample[:min_r_len]
                deg_resample = deg_resample[:min_r_len]
                
                score = pesq_c.pesq(target_sr, ref_resample, deg_resample, mode)
                return round(float(score), 3)
        except Exception as e:
            print(f"Notice: C-extension pesq failed ({str(e)}), falling back to pure Python implementation.")

    # 2. Fallback to native pure-Python ITU-T P.862 implementation
    from .pesq_pure import pesq_pure_python
    try:
        return pesq_pure_python(ref_signal, deg_signal, sr=sr, mode=mode)
    except Exception as e:
        print(f"Warning: PESQ calculation error: {str(e)}")
        return 1.0
