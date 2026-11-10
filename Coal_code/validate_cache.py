#!/usr/bin/env python3


import argparse
from pathlib import Path
import numpy as np

REQUIRED = [
    "paper_version", "git_commit", "git_dirty", "source_sha256",
    "dataset_id", "filename", "train_len", "test_len",
    "window_sizes", "raw_scores_loo", "curv_window_scores_loo",
    "valid_start", "valid_end",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--expected", type=int, default=250)
    a = p.parse_args()
    d = Path(a.root) / f"seed_{a.seed}" / "scores"
    fs = sorted(d.glob("*.npz"))
    if len(fs) != a.expected:
        raise RuntimeError(f"Expected {a.expected} NPZ, found {len(fs)} in {d}")
    ids = []
    for f in fs:
        with np.load(f, allow_pickle=False) as z:
            missing = [k for k in REQUIRED if k not in z]
            if missing: raise RuntimeError(f"{f.name}: missing {missing}")
            r, c = np.asarray(z["raw_scores_loo"]), np.asarray(z["curv_window_scores_loo"])
            if r.shape != c.shape: raise RuntimeError(f"{f.name}: score shape mismatch {r.shape} vs {c.shape}")
            if r.ndim != 2: raise RuntimeError(f"{f.name}: expected 2D scores, got {r.shape}")
            if r.shape[0] != len(np.asarray(z["window_sizes"])): raise RuntimeError(f"{f.name}: window count mismatch")
            if not np.isfinite(r).all() or not np.isfinite(c).all(): raise RuntimeError(f"{f.name}: nonfinite scores")
            ids.append(int(np.asarray(z["dataset_id"]).item()))
    if len(set(ids)) != len(ids): raise RuntimeError("Duplicate dataset IDs in cache")
    print(f"CACHE VALIDATION PASSED: seed={a.seed}, files={len(fs)}")


if __name__ == "__main__": main()
