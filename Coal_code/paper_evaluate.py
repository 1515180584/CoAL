#!/usr/bin/env python3
"""Localize anomalies from cached scores and compute localization metrics."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from src.paper_config import (
    CANDIDATE_PROTOCOL,
    DECISION_PROTOCOL,
    EVENT_LOCALIZATION,
    PAPER_VERSION,
)
from src.pipeline import predict_from_score_curves, candidate_oracle, event_oracle, is_correct


def scalar(z, key, cast=int):
    return cast(np.asarray(z[key]).reshape(()).item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="Root containing seed_0/seed_1/... dirs")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--seeds", nargs="+", type=int, required=True)
    p.add_argument("--expected_datasets", type=int, default=250)
    args = p.parse_args()

    root = Path(args.root).expanduser().resolve()
    out = Path(args.out_dir).expanduser().resolve(); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.seeds:
        score_dir = root / f"seed_{seed}" / "scores"
        paths = sorted(score_dir.glob("*.npz"))
        if len(paths) != args.expected_datasets:
            raise RuntimeError(f"{score_dir}: expected {args.expected_datasets}, found {len(paths)}")
        for path in paths:
            with np.load(path, allow_pickle=False) as z:
                cache_version = scalar(z, "paper_version", str)
                if cache_version != PAPER_VERSION:
                    raise RuntimeError(
                        f"{path.name}: expected score cache {PAPER_VERSION}, got {cache_version}"
                    )
                did = scalar(z, "dataset_id")
                filename = str(np.asarray(z["filename"]).reshape(()).item())
                train_len = scalar(z, "train_len")
                test_len = scalar(z, "test_len")
                true_start, true_end = scalar(z, "true_start"), scalar(z, "true_end")
                valid_start, valid_end = scalar(z, "valid_start"), scalar(z, "valid_end")
                windows = np.asarray(z["window_sizes"], dtype=int)
                raw = np.asarray(z["raw_scores_loo"], dtype=np.float64)
                curv = np.asarray(z["curv_window_scores_loo"], dtype=np.float64)
            result = predict_from_score_curves(raw, curv, windows, train_len)
            cand_or, n_cand = candidate_oracle(
                result["a3_scores"], valid_start, valid_end, train_len, windows
            )
            ev_or, n_ev = event_oracle(result["events"], valid_start, valid_end)
            final_ok = is_correct(result["pred_idx"], valid_start, valid_end)
            rows.append({
                "method": "PAPER_MAINLINE",
                "paper_version": PAPER_VERSION,
                "decision_protocol": DECISION_PROTOCOL,
                "event_localization": EVENT_LOCALIZATION,
                "candidate_protocol": CANDIDATE_PROTOCOL,
                "seed": seed, "dataset_id": did, "filename": filename,
                "train_len": train_len, "test_len": test_len,
                "true_start": true_start, "true_end": true_end,
                "anomaly_start_test": true_start - train_len,
                "anomaly_end_test": true_end - train_len,
                "valid_start": valid_start, "valid_end": valid_end,
                "window_sizes": ";".join(str(int(w)) for w in windows),
                "pred_idx": int(result["pred_idx"]), "final_correct": int(final_ok),
                "hit_legacy": int(final_ok),
                "hit_fixed_100": int(true_start - 100 <= result["pred_idx"] <= true_end + 100),
                "hit_strict": int(true_start <= result["pred_idx"] <= true_end),
                "candidate_oracle_correct": int(cand_or), "event_oracle_correct": int(ev_or),
                "n_correct_candidates": int(n_cand), "n_correct_events": int(n_ev),
                "selected_window": int(result["selected_window"]),
                "event_score": float(result["event_score"]),
                "n_support_windows": int(result["n_support_windows"]),
                "n_event_hypotheses": int(len(result["events"])),
                "event_common_left": float(result["common_left"]),
                "event_common_right": float(result["common_right"]),
            })

    df = pd.DataFrame(rows).sort_values(["seed", "dataset_id"])
    df.to_csv(out / "paper_seed_dataset_results.csv", index=False)
    per = df.groupby("seed").agg(
        datasets=("dataset_id", "count"),
        final_accuracy=("final_correct", "mean"),
        candidate_oracle=("candidate_oracle_correct", "mean"),
        event_oracle=("event_oracle_correct", "mean"),
        mean_support=("n_support_windows", "mean"),
        mean_events=("n_event_hypotheses", "mean"),
    ).reset_index()
    per.to_csv(out / "paper_per_seed_summary.csv", index=False)

    summary = {
        "method": "PAPER_MAINLINE",
        "paper_version": PAPER_VERSION,
        "decision_protocol": DECISION_PROTOCOL,
        "event_localization": EVENT_LOCALIZATION,
        "n_seeds": len(per),
        "final_mean": float(per.final_accuracy.mean()),
        "final_std": float(np.std(per.final_accuracy, ddof=0)),
        "candidate_oracle_mean": float(per.candidate_oracle.mean()),
        "event_oracle_mean": float(per.event_oracle.mean()),
        "mean_support": float(df.n_support_windows.mean()),
        "mean_events": float(df.n_event_hypotheses.mean()),
    }
    pd.DataFrame([summary]).to_csv(out / "paper_across_seed_summary.csv", index=False)

    lines = [
        "=" * 100,
        f"CoAL LOCALIZATION RESULTS ({PAPER_VERSION}; {DECISION_PROTOCOL})",
        "=" * 100,
        "Raw + window-relative curvature -> LOO normal calibration -> co-local A3 ->",
        "complete-window TEST-only Top1 -> common-intersection cross-scale events -> SUM -> intersection-right endpoint",
        "",
        "Per-seed results:",
    ]
    for _, r in per.iterrows():
        lines.append(
            f"  seed={int(r.seed)} Final={100*r.final_accuracy:.2f}% | "
            f"CandOracle={100*r.candidate_oracle:.2f}% | EventOracle={100*r.event_oracle:.2f}% | "
            f"Support={r.mean_support:.3f} | Events={r.mean_events:.3f}"
        )
    lines += [
        "",
        f"Across seeds: Final={100*summary['final_mean']:.2f}% ±{100*summary['final_std']:.2f} | "
        f"CandOracle={100*summary['candidate_oracle_mean']:.2f}% | "
        f"EventOracle={100*summary['event_oracle_mean']:.2f}%",
        "=" * 100,
        "",
    ]
    report = "\n".join(lines)
    (out / "PAPER_MAINLINE_RESULTS.txt").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
