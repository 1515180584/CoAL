"""UCR loading, window construction, differences and score alignment."""

from pathlib import Path
import numpy as np
from scipy.signal import correlate, find_peaks

from .paper_config import (
    WINDOW_LOWER_BOUND, NUM_WINDOW_CANDIDATES, DEFAULT_ACF_FALLBACK,
    ACF_MIN_LAG, ACF_MAX_LAG, ACF_PEAK_DISTANCE,
)


def parse_ucr_filename(filename):
    """Parse dataset metadata from a UCR anomaly filename."""
    stem = Path(filename).stem
    parts = stem.split("_")

    if len(parts) < 4 or not parts[0].isdigit():
        raise ValueError(f"Unexpected UCR filename: {filename}")
    return int(parts[0]), int(parts[-3]), int(parts[-2]), int(parts[-1])


def load_ucr_compatible(path):
    """Min-max scale the complete series, then split at the filename training boundary. Test extrema participate in this offline normalization."""

    temp = np.loadtxt(path).reshape(-1)
    did, train_end, true_start, true_end = parse_ucr_filename(Path(path).name)


    lo, hi = float(np.min(temp)), float(np.max(temp))
    denom = hi - lo if hi != lo else 1.0
    temp = (temp - lo) / denom


    train = temp[:train_end].reshape(-1)
    test = temp[train_end:].reshape(-1)
    return did, train, test, true_start, true_end


def calculate_window_upper_bound(train):
    """Return the first raw-correlation peak lag in [10,1000], or 80 if none exists."""
    data = np.asarray(train).reshape(-1)

    autocor = correlate(data, data, mode="full")
    autocor = autocor[len(autocor) // 2:]


    peak_range = np.arange(ACF_MIN_LAG, min(len(autocor), ACF_MAX_LAG + 1))
    peaks, _ = find_peaks(autocor[peak_range], distance=ACF_PEAK_DISTANCE)

    peaks = peak_range[peaks]
    return int(np.floor(peaks[0])) if len(peaks) > 0 else DEFAULT_ACF_FALLBACK


def candidate_window_sizes(upper_bound):
    """Generate equally spaced candidate window lengths between the fixed lower bound and estimated upper bound."""
    lower = WINDOW_LOWER_BOUND
    n = NUM_WINDOW_CANDIDATES
    increment = (int(upper_bound) - lower) // (n - 1)
    return np.unique([lower + i * increment for i in range(n)]).astype(np.int32)


def create_windows(data, window_size):
    """Construct all contiguous sliding windows from a one-dimensional sequence."""
    x = np.asarray(data).reshape(-1)
    w = int(window_size)
    if w <= 0 or w > len(x):
        raise ValueError(f"Invalid window_size={w} for length={len(x)}")
    return np.lib.stride_tricks.sliding_window_view(x, window_shape=w)


def window_curvature(windows):
    """Apply centered second differences independently within each reflect-padded window."""
    x = np.asarray(windows, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] < 2:
        raise ValueError(f"Bad window matrix for curvature: {x.shape}")

    p = np.pad(x, ((0, 0), (1, 1)), mode="reflect")

    return p[:, 2:] - 2.0 * p[:, 1:-1] + p[:, :-2]


def pad_window_scores(scores, window_size):
    """Right-align window scores by prepending window_size-1 zeros."""
    return np.concatenate([
        np.zeros(int(window_size) - 1, dtype=np.float64),
        np.asarray(scores, dtype=np.float64),
    ])


def ucr_valid_interval(true_start, true_end):
    """Expand the labeled anomaly interval using the Legacy evaluation tolerance."""
    tolerance = max(100, int(true_end) - int(true_start))
    return int(true_start) - tolerance, int(true_end) + tolerance, tolerance
