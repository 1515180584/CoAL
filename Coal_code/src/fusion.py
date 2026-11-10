"""Train-reference calibration and co-local dual-view evidence fusion."""

import numpy as np
from scipy.ndimage import maximum_filter1d
from .calibration import empirical_tail_evidence


def local_max_segmented(x, radius, train_len):
    """Compute radius-based maxima separately within train and test segments."""
    x = np.asarray(x, dtype=np.float64)
    radius = max(0, int(radius))

    if radius == 0:
        return x.copy()
    size = 2 * radius + 1
    out = np.empty_like(x)

    split = int(np.clip(train_len, 0, len(x)))
    if split > 0:
        out[:split] = maximum_filter1d(x[:split], size=size, mode="constant", cval=0.0)
    if split < len(x):
        out[split:] = maximum_filter1d(x[split:], size=size, mode="constant", cval=0.0)
    return out


def make_a3_colocal(raw_full, curv_full, window, train_len):
    """Calibrate each view on complete training windows; return M + (1-M)G, with M=max(R,D2) and G the segmented radius-w maximum of min(R,D2)."""

    train_start = max(0, int(window) - 1)
    if train_start >= int(train_len):
        raise ValueError(f"window={window} leaves no train reference for train_len={train_len}")


    raw_ref = np.asarray(raw_full[train_start:train_len], dtype=np.float64)
    curv_ref = np.asarray(curv_full[train_start:train_len], dtype=np.float64)
    a_raw = empirical_tail_evidence(raw_full, raw_ref)
    a_curv = empirical_tail_evidence(curv_full, curv_ref)


    strongest_view = np.maximum(a_raw, a_curv)
    same_location_joint = np.minimum(a_raw, a_curv)

    local_agreement = local_max_segmented(same_location_joint, int(window), train_len)

    a3 = strongest_view + (1.0 - strongest_view) * local_agreement
    return np.clip(a3, 0.0, 1.0)
