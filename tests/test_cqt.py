import numpy as np
from core.cqt.logfmap import compute_logfmap
from core.cqt.forward import linear_to_cqt
from core.cqt.inverse import cqt_to_linear

def test_cqt_logfmap():
    I = 513
    low_row = 10
    M, N, opr = compute_logfmap(I=I, L=low_row, H=I)
    
    assert M.shape == (opr, I)
    assert N.shape == (I, opr)
    assert opr > 0

def test_cqt_roundtrip_transform():
    linear_sgram = np.random.uniform(0, 10, (513, 100)).astype(np.float32)
    
    cqt_sgram, meta = linear_to_cqt(linear_sgram, n_fft=1024, low_row=10, cqt_bins=513)
    assert cqt_sgram.shape == (513, 100)
    
    recon_linear = cqt_to_linear(cqt_sgram, target_bins=513, low_row=10, metadata=meta)
    assert recon_linear.shape == (513, 100)

if __name__ == "__main__":
    test_cqt_logfmap()
    test_cqt_roundtrip_transform()
    print("test_cqt passed!")
