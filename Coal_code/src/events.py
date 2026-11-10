"""Construct and rank common-intersection events from scale-wise candidates."""

from dataclasses import dataclass, replace
from typing import Dict, List, Sequence, Tuple
import numpy as np


def extract_top1_test_only(scores, train_len, window, candidate_offset=None):
    """Return the earliest maximum at complete test-window endpoints (zero-based start: train_len + window - 1). candidate_offset optionally overrides the test-local start."""
    scores = np.asarray(scores, dtype=np.float64)
    window = int(window)
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}.")


    if candidate_offset is None:
        candidate_offset = window - 1
    candidate_offset = int(candidate_offset)
    if candidate_offset < 0:
        raise ValueError(f"candidate_offset must be non-negative, got {candidate_offset}.")
    start = int(np.clip(int(train_len) + candidate_offset, 0, len(scores)))
    if start >= len(scores):
        raise ValueError(
            f"No complete test window available: train_len={train_len}, "
            f"window={window}, time={len(scores)}."
        )
    region = scores[start:]
    finite = np.isfinite(region)
    if not finite.any():
        raise ValueError("No finite score in test candidate region.")

    safe = np.where(finite, region, -np.inf)
    local_idx = int(np.argmax(safe))

    idx = start + local_idx
    return {"idx": idx, "score": float(scores[idx])}


def weighted_median(values, weights):
    """Compute the weighted median of one-dimensional values and round it to an integer time index."""
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)

    order = np.argsort(v)
    v, w = v[order], w[order]
    total = float(w.sum())
    if not np.isfinite(total) or total <= 0:
        return int(round(float(np.median(v))))

    j = int(np.searchsorted(np.cumsum(w), 0.5 * total, side="left"))
    return int(round(float(v[min(j, len(v) - 1)])))


@dataclass(frozen=True)
class Candidate:
    """The highest-scoring test candidate contributed by one window scale."""
    cid: int
    source_window_index: int
    source_window_rank: int
    source_window: int
    idx: int
    evidence: float
    left: float = 0.0
    right: float = 0.0

    def with_end_aligned_interval(self):
        """Return a new candidate with its interval set to ``[idx-window+1, idx]``."""
        p = float(self.idx)
        w = float(self.source_window)

        return replace(self, left=p - w + 1.0, right=p)


def build_top1_candidate_pool(a3_scores, windows, train_len, candidate_start_offsets=None):
    """Build one highest-scoring candidate per window scale using only the test segment."""
    if len(a3_scores) != len(windows):
        raise ValueError("a3_scores and windows have different lengths.")
    if candidate_start_offsets is None:
        offsets = [None] * len(windows)
    else:
        offsets = np.asarray(candidate_start_offsets, dtype=int).reshape(-1)
        if len(offsets) != len(windows):
            raise ValueError("candidate_start_offsets and windows have different lengths.")
    pool = []
    for i, (scores, w, offset) in enumerate(zip(a3_scores, windows, offsets)):
        c = extract_top1_test_only(scores, train_len, int(w), candidate_offset=offset)

        pool.append(Candidate(
            cid=i,
            source_window_index=i,
            source_window_rank=i + 1,
            source_window=int(w),
            idx=int(c["idx"]),
            evidence=float(c["score"]),
        ).with_end_aligned_interval())
    return pool


def _probe_points(candidates):
    """Generate probe points sufficient to enumerate all sets of simultaneously active intervals."""
    endpoints = sorted({float(c.left) for c in candidates} | {float(c.right) for c in candidates})
    if not endpoints:
        return []
    probes = set(endpoints)
    for a, b in zip(endpoints[:-1], endpoints[1:]):
        if a < b:
            probes.add(0.5 * (a + b))
    return sorted(probes)


def _active_at(candidates, point):
    """Return all candidates whose inclusive intervals contain the probe point."""

    eps = 1e-12
    return [c for c in candidates if (c.left - eps) <= point <= (c.right + eps)]


def _event_key(reps):
    """Build an order-independent event membership key from scale ranks and peak positions."""
    return tuple(sorted((c.source_window_rank, c.idx) for c in reps))


def _summarize_event(event_id, reps, anchor):
    """Summarize candidates with a common intersection as an event dictionary."""

    reps = sorted(reps, key=lambda c: c.source_window_index)
    evidences = np.asarray([c.evidence for c in reps], dtype=np.float64)


    common_left = float(max(c.left for c in reps))
    common_right = float(min(c.right for c in reps))
    pred_idx = int(round(common_right))
    return {
        "event_id": int(event_id),

        "event_score": float(evidences.sum()),
        "n_support_windows": int(len(reps)),
        "pred_idx": int(pred_idx),

        "common_left": common_left,
        "common_right": common_right,

        "anchor": float(anchor),
        "representatives": list(reps),

        "max_evidence": float(evidences.max()),
        "mean_evidence": float(evidences.mean()),
    }


def cross_scale_event_consensus(
    a3_scores, windows, train_len, candidate_start_offsets=None
):
    """Return maximal common-intersection events and per-scale Top1 candidates."""
    candidates = build_top1_candidate_pool(
        a3_scores,
        windows,
        train_len,
        candidate_start_offsets=candidate_start_offsets,
    )

    by_key: Dict[Tuple[Tuple[int, int], ...], dict] = {}
    next_event_id = 1
    for point in _probe_points(candidates):
        active = _active_at(candidates, point)
        if not active:
            continue
        key = _event_key(active)
        event = _summarize_event(next_event_id, active, point)
        if key not in by_key:
            by_key[key] = event
            next_event_id += 1
        else:


            old = by_key[key]
            mid = 0.5 * (event["common_left"] + event["common_right"])
            if abs(event["anchor"] - mid) < abs(old["anchor"] - mid):
                event["event_id"] = old["event_id"]
                by_key[key] = event

    events = list(by_key.values())


    rep_sets = [{(c.source_window_rank, c.idx) for c in e["representatives"]} for e in events]
    keep = []
    for i, event in enumerate(events):
        s = rep_sets[i]
        if not any(i != j and s < rep_sets[j] for j in range(len(events))):
            keep.append(event)

    keep.sort(key=lambda e: (e["pred_idx"], -e["event_score"]))
    for i, event in enumerate(keep, start=1):
        event["event_id"] = i
    if not keep:
        raise RuntimeError("No valid cross-scale event hypothesis.")
    return keep, candidates


def select_best_event(events):
    """Rank by summed evidence, support count, maximum member evidence, then earlier intersection-right position."""
    if not events:
        raise RuntimeError("No valid event hypothesis.")

    best = max(events, key=lambda e: (
        e["event_score"],
        e["n_support_windows"],
        e["max_evidence"],
        -e["pred_idx"],
    ))

    representative = min(
        best["representatives"],
        key=lambda c: (abs(c.idx - best["pred_idx"]), -c.evidence),
    )
    return {

        "pred_idx": int(best["pred_idx"]),
        "selected_window_rank": int(representative.source_window_rank),
        "selected_window": int(representative.source_window),
        "event_score": float(best["event_score"]),
        "n_support_windows": int(best["n_support_windows"]),
        "event_id": int(best["event_id"]),
        "common_left": float(best["common_left"]),
        "common_right": float(best["common_right"]),
    }
