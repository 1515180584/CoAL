#!/usr/bin/env python3
"""Generate multi-scale Raw/D2 nearest-neighbor score curves."""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse, csv, gc, hashlib, re, subprocess, sys, time, traceback
from functools import lru_cache
from pathlib import Path
import numpy as np

from src.paper_config import PAPER_VERSION
from src.ucr_utils import (
    parse_ucr_filename, load_ucr_compatible, calculate_window_upper_bound,
    candidate_window_sizes, create_windows, window_curvature, pad_window_scores,
    ucr_valid_interval,
)
from src.random_scoring import derive_seed, fit_score_branch_loo


def args_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_folder", default="data/UCR")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_jobs", type=int, default=10)
    p.add_argument("--knn_native_threads", type=int, default=10)
    p.add_argument("--ids_file")
    p.add_argument("--max_files", type=int)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--summary_name", default="generation_summary.csv")
    return p


def requested_ids(path):
    if not path:
        return None
    tokens = re.findall(r"\d+", Path(path).read_text(encoding="utf-8"))
    if not tokens:
        raise ValueError(f"No dataset IDs in {path}")
    return {int(x) for x in tokens}


@lru_cache(maxsize=1)
def git_state():
    code_dir = Path(__file__).resolve().parent
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(code_dir), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(code_dir), "status", "--porcelain", "--untracked-files=all"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return commit, bool(status.strip())
    except Exception:
        return "unknown", True


@lru_cache(maxsize=1)
def source_sha256():
    """Fingerprint the executable final-method source, including uncommitted files."""
    code_dir = Path(__file__).resolve().parent
    paths = [
        code_dir / "paper_generate_scores.py",
        code_dir / "paper_evaluate.py",
        code_dir / "validate_cache.py",
    ]
    paths.extend(sorted((code_dir / "src").glob("*.py")))
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda p: p.relative_to(code_dir).as_posix()):
        relative = path.relative_to(code_dir).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def process_one(path, args, score_dir):
    did, train, test, true_start, true_end = load_ucr_compatible(path)
    train_len, test_len = len(train), len(test)
    dataset_seed = derive_seed(args.seed, did, 0, 0, 0)
    np.random.seed(dataset_seed)

    upper = calculate_window_upper_bound(train)
    windows = candidate_window_sizes(upper)
    if len(windows) == 0:
        raise RuntimeError("No candidate windows generated.")

    raw_all, curv_all = [], []
    raw_meta, curv_meta = [], []
    for rank0, w0 in enumerate(windows):
        w, rank = int(w0), rank0 + 1
        raw_seed = derive_seed(args.seed, did, rank, 1, 1)
        raw_kss = derive_seed(args.seed, did, rank, 1, 2)
        curv_seed = derive_seed(args.seed, did, rank, 2, 1)
        curv_kss = derive_seed(args.seed, did, rank, 2, 2)

        train_w = create_windows(train, w)
        test_w = create_windows(test, w)
        r_train, r_test, r_info = fit_score_branch_loo(
            train_w, test_w, raw_seed, raw_kss,
            n_jobs=args.n_jobs, knn_native_threads=args.knn_native_threads,
        )
        train_c = window_curvature(train_w)
        test_c = window_curvature(test_w)
        c_train, c_test, c_info = fit_score_branch_loo(
            train_c, test_c, curv_seed, curv_kss,
            n_jobs=args.n_jobs, knn_native_threads=args.knn_native_threads,
        )

        raw_all.append(np.concatenate([pad_window_scores(r_train, w), pad_window_scores(r_test, w)]))
        curv_all.append(np.concatenate([pad_window_scores(c_train, w), pad_window_scores(c_test, w)]))
        raw_meta.append(r_info); curv_meta.append(c_info)
        del train_w, test_w, train_c, test_c
        gc.collect()

    valid_start, valid_end, tolerance = ucr_valid_interval(true_start, true_end)
    commit, dirty = git_state()
    code_hash = source_sha256()
    payload = {
        "paper_version": np.asarray(PAPER_VERSION),
        "git_commit": np.asarray(commit),
        "git_dirty": np.asarray(dirty, dtype=np.bool_),
        "source_sha256": np.asarray(code_hash),
        "dataset_id": np.asarray(did, dtype=np.int32),
        "filename": np.asarray(Path(path).name),
        "base_seed": np.asarray(int(args.seed), dtype=np.int64),
        "dataset_seed": np.asarray(int(dataset_seed), dtype=np.uint32),
        "train_len": np.asarray(train_len, dtype=np.int64),
        "test_len": np.asarray(test_len, dtype=np.int64),
        "true_start": np.asarray(true_start, dtype=np.int64),
        "true_end": np.asarray(true_end, dtype=np.int64),
        "valid_start": np.asarray(valid_start, dtype=np.int64),
        "valid_end": np.asarray(valid_end, dtype=np.int64),
        "tolerance_L": np.asarray(tolerance, dtype=np.int64),
        "window_sizes": np.asarray(windows, dtype=np.int32),
        "raw_scores_loo": np.stack(raw_all).astype(np.float32, copy=False),
        "curv_window_scores_loo": np.stack(curv_all).astype(np.float32, copy=False),
        "raw_mean_train_loo": np.asarray([m["mean_train_loo"] for m in raw_meta]),
        "curv_mean_train_loo": np.asarray([m["mean_train_loo"] for m in curv_meta]),
    }
    for j in range(len(windows)):
        payload[f"raw_mask_w{j+1}"] = raw_meta[j]["mask"]
        payload[f"curv_window_mask_w{j+1}"] = curv_meta[j]["mask"]

    out_path = score_dir / f"dataset_{did:03d}_seed_{int(args.seed)}.npz"
    np.savez_compressed(out_path, **payload)
    return {
        "dataset_id": did,
        "filename": Path(path).name,
        "seed": int(args.seed),
        "train_len": train_len,
        "test_len": test_len,
        "window_sizes": ";".join(map(str, windows.tolist())),
        "paper_version": PAPER_VERSION,
        "git_commit": commit,
        "git_dirty": dirty,
        "source_sha256": code_hash,
        "status": "ok",
        "error": "",
    }


def write_summary(path, rows):
    if not rows:
        return
    fields = list(rows[0].keys())
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def main():
    args = args_parser().parse_args()
    if args.resume and args.overwrite:
        raise ValueError("Use only one of --resume/--overwrite.")
    root = Path(args.dataset_folder).expanduser().resolve()
    out = Path(args.output_dir).expanduser().resolve()
    score_dir = out / "scores"; score_dir.mkdir(parents=True, exist_ok=True)
    summary_dir = out / "generation_summaries"; summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / args.summary_name

    if args.overwrite:
        for p in score_dir.glob("*.npz"): p.unlink()
        if summary_path.exists(): summary_path.unlink()

    ids = requested_ids(args.ids_file)
    files = []
    for p in root.glob("*.txt"):
        try: did, _, _, _ = parse_ucr_filename(p.name)
        except Exception: continue
        if ids is None or did in ids: files.append((did, p))
    files.sort(key=lambda x: x[0])
    if args.max_files is not None: files = files[:args.max_files]

    print("=" * 100)
    print(f"{PAPER_VERSION}: Raw + window-curvature + LOO + co-local A3 + cross-scale event consensus")
    print(f"Python={sys.version.split()[0]} seed={args.seed} datasets={len(files)}")
    print("Method parameters are defined in src/paper_config.py.")
    print("=" * 100)

    rows = []
    for pos, (did, p) in enumerate(files, 1):
        target = score_dir / f"dataset_{did:03d}_seed_{int(args.seed)}.npz"
        if args.resume and target.exists():
            print(f"[{pos:03d}/{len(files):03d}] ID={did:03d} SKIP")
            continue
        t0 = time.time()
        try:
            row = process_one(p, args, score_dir)
            row["runtime_sec"] = round(time.time() - t0, 3)
            rows.append(row)
            print(f"[{pos:03d}/{len(files):03d}] ID={did:03d} OK {row['runtime_sec']:.1f}s")
        except Exception as exc:
            traceback.print_exc()
            rows.append({
                "dataset_id": did, "filename": p.name, "seed": args.seed,
                "train_len": "", "test_len": "", "window_sizes": "",
                "paper_version": PAPER_VERSION,
                "git_commit": git_state()[0],
                "git_dirty": git_state()[1],
                "source_sha256": source_sha256(),
                "status": "error", "error": f"{type(exc).__name__}: {exc}",
                "runtime_sec": round(time.time() - t0, 3),
            })
            print(f"[{pos:03d}/{len(files):03d}] ID={did:03d} ERROR: {exc}")
        gc.collect()
    write_summary(summary_path, rows)
    print(f"Scores: {score_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
