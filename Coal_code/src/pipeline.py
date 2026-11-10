"""Predict an anomaly location from multi-scale Raw/D2 scores."""

import numpy as np
from .fusion import make_a3_colocal
from .events import extract_top1_test_only, cross_scale_event_consensus, select_best_event


def is_correct(idx, valid_start, valid_end):
    """Return 1 if the global sequence index lies within the inclusive evaluation interval, otherwise 0."""
    return int(int(valid_start) <= int(idx) <= int(valid_end))


def predict_from_score_curves(raw_scores, curv_scores, windows, train_len):
    """Calibrate and fuse Raw/D2 curves, construct maximal events, and return the winning intersection-right location. Inputs are [scales,time] score matrices, window lengths and train_len; labels are not used."""

    raw_scores = np.asarray(raw_scores, dtype=np.float64)
    curv_scores = np.asarray(curv_scores, dtype=np.float64)
    windows = np.asarray(windows, dtype=int)
    if raw_scores.ndim != 2 or curv_scores.ndim != 2:
        raise ValueError("raw_scores and curv_scores must be [n_windows, time].")
    if raw_scores.shape != curv_scores.shape:
        raise ValueError("Raw and Curvature score arrays must have same shape.")
    if raw_scores.shape[0] != len(windows):
        raise ValueError("Number of score curves does not match window count.")


    a3_scores = [
        make_a3_colocal(raw_scores[i], curv_scores[i], int(w), int(train_len))
        for i, w in enumerate(windows)
    ]


    events, candidates = cross_scale_event_consensus(a3_scores, windows, int(train_len))
    selected = select_best_event(events)


    if any(c.idx < int(train_len) + int(c.source_window) - 1 for c in candidates):
        raise AssertionError("Complete-test-window invariant violated by a candidate.")
    if selected["pred_idx"] < int(train_len):
        raise AssertionError("TEST-ONLY invariant violated: selected prediction in train region.")
    if not (
        selected["common_left"] <= selected["pred_idx"] <= selected["common_right"]
    ):
        raise AssertionError("Selected prediction is outside the event common intersection.")
    if selected["pred_idx"] != int(round(selected["common_right"])):
        raise AssertionError("Intersection-right localization invariant violated.")
    return {**selected, "a3_scores": a3_scores, "events": events, "candidates": candidates}


def candidate_oracle(a3_scores, valid_start, valid_end, train_len, windows):
    """Evaluate whether any scale-wise candidate hits the labeled interval; evaluation only."""
    if len(a3_scores) != len(windows):
        raise ValueError("a3_scores and windows have different lengths.")
    oks = []
    for scores, window in zip(a3_scores, windows):
        c = extract_top1_test_only(scores, train_len, int(window))
        oks.append(is_correct(c["idx"], valid_start, valid_end))
    return int(any(oks)), int(sum(oks))


def event_oracle(events, valid_start, valid_end):
    """Evaluate whether any event position hits the labeled interval; evaluation only."""
    oks = [is_correct(e["pred_idx"], valid_start, valid_end) for e in events]
    return int(any(oks)), int(sum(oks))
