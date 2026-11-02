"""Light audio helpers with no heavy imports (the build checks them against librosa)."""
import math

import numpy as np
import soxr


def resample16(y, sr):
    """Mono float32 → 16 kHz, bit-identical to librosa.resample(res_type="soxr_hq", fix=True), without librosa: its first call
    in a fresh worker costs numba compilation (measured: 14.5 s on a worker CPU for 0.15 s of actual resampling)."""
    if sr == 16000:
        return y
    n = int(math.ceil(len(y) * 16000 / sr))
    z = soxr.resample(y, sr, 16000, quality="soxr_hq")
    z = z[:n] if len(z) >= n else np.pad(z, (0, n - len(z)))
    return np.asarray(z, dtype=np.float32)
