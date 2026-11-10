"""MiniRocket/KSS representations and leave-one-out nearest-neighbor scores."""
import os
import sys
from pathlib import Path
import numpy as np
from joblib import parallel_backend
from threadpoolctl import threadpool_limits
from sklearn.neighbors import NearestNeighbors

from .paper_config import (
    N_KERNEL, KSS_SAMPLES, KSS_KEEP_RATE, KSS_ALPHA, KSS_BETA, K_NEIGHBORS,
)


def _load_randomad_dependencies():
    """Load bundled dependencies, or dependencies specified by RANDOMAD_ROOT."""


    configured = os.environ.get("RANDOMAD_ROOT")
    root = (
        Path(configured).expanduser().resolve()
        if configured
        else Path(__file__).resolve().parent
    )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:

        from minirocket import transform
        from randomad_core import MiniRocket_fit, RandomAD_mask, remove_masked_columns
    except Exception as exc:

        raise ImportError(
            "Could not import minirocket.py and randomad_core.py. Point "
            "RANDOMAD_ROOT at RandomAD_final/src (or another verified dependency "
            "root whose RandomAD_mask accepts the n_jobs runtime argument)."
        ) from exc
    return transform, MiniRocket_fit, RandomAD_mask, remove_masked_columns


def derive_seed(base_seed, dataset_id, window_rank, view_code, stage_code):
    """Derive a 32-bit seed from dataset, scale, view and operation identifiers."""

    x = int(base_seed) & 0xFFFFFFFF
    x = (x * 1000003 + int(dataset_id) * 9176 + int(window_rank) * 101
         + int(view_code) * 10007 + int(stage_code) * 10000019 + 12345)

    return int(x % (2 ** 32 - 1))


def loo_scores_from_neighbors(distances, indices, k):
    """Remove each query sample by index from its training-neighbor results and compute leave-one-out scores."""
    n = distances.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        d = np.asarray(distances[i], dtype=np.float64)
        idx = np.asarray(indices[i], dtype=np.int64)
        pos = np.flatnonzero(idx == i)
        if pos.size:

            keep = np.ones(len(idx), dtype=bool)
            keep[int(pos[0])] = False
            vals = d[keep][:k]
        else:


            vals = d[:k]
        if len(vals) != k:

            raise RuntimeError(f"LOO extraction failed at row {i}: got {len(vals)}")
        out[i] = float(np.mean(vals))
    return out


def fit_score_branch_loo(
    train_windows,
    test_windows,
    rocket_seed,
    kss_seed,
    n_jobs=10,
    knn_native_threads=10,
):
    """Fit MiniRocket/KSS; return train LOO scores, test-to-train scores and metadata."""
    transform, MiniRocket_fit, RandomAD_mask, remove_masked_columns = _load_randomad_dependencies()


    train_windows = np.require(train_windows, dtype=np.float32, requirements=["C", "A", "W"])
    test_windows = np.require(test_windows, dtype=np.float32, requirements=["C", "A", "W"])


    np.random.seed(int(rocket_seed))
    para = MiniRocket_fit(train_windows, num_features=N_KERNEL)


    np.random.seed(int(kss_seed))

    with parallel_backend("loky", inner_max_num_threads=1):
        mask = RandomAD_mask(
            para,
            train_windows,
            keep_features_ratio=KSS_KEEP_RATE,
            num_samples=KSS_SAMPLES,
            alpha=KSS_ALPHA,
            beta=KSS_BETA,
            n_jobs=int(n_jobs),
        )


    train_features = remove_masked_columns(transform(train_windows, para), mask)
    test_features = remove_masked_columns(transform(test_windows, para), mask)

    k = K_NEIGHBORS

    if len(train_features) <= k:
        raise ValueError(f"Need at least k+1 train windows: n={len(train_features)}, k={k}")


    with threadpool_limits(limits=max(1, int(knn_native_threads))):
        nbrs = NearestNeighbors(algorithm="auto", n_jobs=1).fit(train_features)

        d_loo, i_loo = nbrs.kneighbors(train_features, n_neighbors=k + 1)
        train_loo = loo_scores_from_neighbors(d_loo, i_loo, k)

        d_test, _ = nbrs.kneighbors(test_features, n_neighbors=k)
        test_scores = d_test.mean(axis=1)


    info = {
        "mask": np.asarray(mask).copy(),
        "mask_length": int(len(mask)),
        "selected_features": int(np.count_nonzero(mask)),
        "rocket_seed": int(rocket_seed),
        "kss_seed": int(kss_seed),
        "mean_train_loo": float(np.mean(train_loo)),
        "median_train_loo": float(np.median(train_loo)),
    }
    return train_loo, test_scores, info
