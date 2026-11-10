# CoAL

Core implementation of **CoAL: Cross-Scale Time Series Anomaly Localization with
Co-Localized Dual-View Evidence**.

Raw and window-relative second differences → MiniRocket/KSS → normal-reference
kNN scores → evidence calibration → co-local A3 fusion → cross-scale
common-intersection events → right-endpoint localization.

## Installation

Use Python 3.10 or 3.11 in a dedicated environment:

```bash
python -m pip install -r requirements.txt
```

## Data

Download the [UCR anomaly archive](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCR_TimeSeriesAnomalyDatasets2021.zip)
and place the 250 original anomaly-series `.txt` files directly in `data/UCR/`.
Preserve their original filenames. Data is not bundled.

## Run

Run from this directory. Clear any existing `RANDOMAD_ROOT` override to use the
bundled implementation. Generate and validate scores for seed 0:

```bash
python paper_generate_scores.py --dataset_folder data/UCR --output_dir results/seed_0 --seed 0 --n_jobs 1 --knn_native_threads 1
python validate_cache.py --root results --seed 0 --expected 250
```

Repeat for seeds 1 and 2 using `results/seed_1` and `results/seed_2`, then evaluate:

```bash
python paper_evaluate.py --root results --out_dir results/evaluation --seeds 0 1 2 --expected_datasets 250
```

For a quick engineering run, add `--max_files 1` to generation, use `--expected 1`
for validation, and evaluate with `--seeds 0 --expected_datasets 1`.
Scientific parameters are defined in `src/paper_config.py`. Output `pred_idx` is a
zero-based global array index. Labels are read for evaluation, not prediction.

This repository contains the core method and primary localization evaluation.
Baseline and ablation experiments are not included. The method uses randomized
representations; results can vary between runs.

## Acknowledgments

The bundled core uses RandomAD by Wenjie Xi and Jessica Lin
([original repository](https://github.com/Jackxiini/RandomAD), ECML PKDD 2025)
and MiniRocket by Angus Dempster, Daniel F. Schmidt and Geoffrey I. Webb
([paper](https://arxiv.org/abs/2012.08791)). Original MiniRocket attribution is
retained in the source. Applicable upstream redistribution terms still require
verification; no blanket license is specified.
