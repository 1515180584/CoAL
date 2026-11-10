"""Map normal-reference distances to bounded anomaly evidence."""

import numpy as np
from .paper_config import TAIL_Q


def empirical_tail_evidence(full_scores, train_ref, tail_q=TAIL_Q):
    """Calibrate using the training empirical right tail and exponential excess above Q95; map -log(p) to bounded evidence e/(1+e)."""

    x = np.asarray(full_scores, dtype=np.float64)
    ref = np.asarray(train_ref, dtype=np.float64)

    ref = ref[np.isfinite(ref)]
    if ref.size == 0:
        raise ValueError("Empty training reference.")


    tail_q = float(np.clip(tail_q, 0.5, 0.9999))


    ref_sorted = np.sort(ref)
    left = np.searchsorted(ref_sorted, x, side="left")
    count_ge = ref_sorted.size - left
    p_emp = (1.0 + count_ge.astype(np.float64)) / (ref_sorted.size + 1.0)


    u = float(np.quantile(ref, tail_q))
    excess = ref[ref > u] - u
    positive = excess[excess > 0]
    if positive.size:

        scale = float(np.mean(positive))
    else:

        q99 = float(np.quantile(ref, min(0.999, max(tail_q, 0.99))))
        scale = q99 - u
    if not np.isfinite(scale) or scale <= 1e-12:

        scale = float(np.std(ref))
    if not np.isfinite(scale) or scale <= 1e-12:

        scale = max(abs(u), 1.0) * 1e-6


    tail_mass = max(float(np.mean(ref >= u)), 1.0 / (ref.size + 1.0))
    y = np.maximum(x - u, 0.0)

    p_tail = tail_mass * np.exp(-np.minimum(y / scale, 700.0))

    p = np.where(x > u, p_tail, p_emp)
    p = np.clip(p, np.finfo(np.float64).tiny, 1.0)


    surprise = -np.log(p)
    evidence = surprise / (1.0 + surprise)

    evidence[~np.isfinite(evidence)] = 0.0
    return np.clip(evidence, 0.0, 1.0)
