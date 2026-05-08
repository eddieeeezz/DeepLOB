# DeepLOB — repository architecture

This document is the single map for **where things live** and **how data flows**. Training notebooks live under `flow_process/`; inference/backtest scripts live at the repo root.

## Layer overview

```text
┌─────────────────────────────────────────────────────────────────┐
│  DeepLOB_data_process/                                           │
│  raw_data/*.csv  ─►  prepare_deeplob_data  |  prepare_multik_export │
│                      ─►  (generated parquet/csv — see .gitignore)   │
│                      ─►  export_test_xy  ─►  copy to repo root      │
└───────────────────────────────┬─────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  Repo root                                                       │
│  test_Xy.parquet | test_10.parquet | test_50.parquet            │
└───────────────┬─────────────────────────┬─────────────────────┘
                ▼                         ▼
     flow_process/*.ipynb          generate_predictions.py
     (train / eval splits)         ─► predictions_*.parquet
                │                         │
                ▼                         ▼
     Output_models/*.pth            backtest_signals.py
                                   ─► console metrics (optional CSV / PNG flags)
```

## Folders

| Path | Role |
|------|------|
| **`DeepLOB_data_process/`** | **Only ETL**: merge raw LOB CSVs, compute labels (single‑k or multi‑k), export slim `index + 20 features + label` parquet. Does **not** host canonical checkpoints (those are under `Output_models/`). |
| **`DeepLOB_data_process/raw_data/`** | Source snapshots (one file per session/day). Required input for `prepare_*.py`. |
| **`flow_process/`** | Canonical **Jupyter training** pipelines for base / k10 / k50 datasets. |
| **`deeplob_eval/`** | Shared Python package (`DeepLOB5Stable`, data prep, evaluation helpers) imported by `DeepLOB_evaluate_trained_models.ipynb`. |
| **`Output_models/`** | Best `.pth` checkpoints aligned with root parquets (`deeplob5_stabilized_best*.pth`). |

## Scripts (repo root)

| Script | Role |
|--------|------|
| **`generate_predictions.py`** | Load `Output_models/*.pth`, run inference on tail slice of a parquet, write **`predictions_*.parquet`** (optional softmax gating on ±1). |
| **`backtest_signals.py`** | Join LOB with predictions; spread vs mid‑to‑mid PnL; **prints stats only** unless `--equity-csv`, `--trades-csv`, or `--plot` is passed. |

## Scripts (`DeepLOB_data_process/`)

Only **merge → label → parquet** tooling remains:

| Script | Role |
|--------|------|
| **`prepare_deeplob_data.py`** | Merge `raw_data/*.csv` → labeled **`test.csv` / `test.parquet`** for one horizon \(k\) (paths configurable in script). |
| **`prepare_multik_export.py`** | Merge all CSV → **`test_multik.parquet`** with `label_k10`, `label_k20`, `label_k50` (+ continuous vol‑weighted returns). |
| **`export_test_xy.py`** | Strip any wide parquet to **`index` + 20 LOB columns + `label`** for notebooks / root naming (`test_Xy.parquet`, etc.). |

Inspect/plot utilities and the standalone **`train_deeplob.py`** trainer were removed; training lives under **`flow_process/`** notebooks and **`Output_models/`**.

## Label idea (short)

LOB labels use **smoothed mid‑price moves** over horizon \(k\) and **intraday volatility normalization** (e.g. 30‑minute bins, rolling prior‑day volatility) before thresholding into **−1 / 0 / +1**. Details match **`prepare_deeplob_data.py`** / milestone PDF in `DeepLOB_data_process/`.

## What was intentionally removed (cleanup)

To reduce redundancy, the repo **no longer** carries under `DeepLOB_data_process/`:

- Plot / inspect helpers (`plot_*.py`, `inspect_test_parquet.py`) and standalone **`train_deeplob.py`** under **`DeepLOB_data_process/`** — training uses **`flow_process/`** notebooks only.
- JetBrains **`.idea/`**, **`__pycache__/`**, notebook checkpoints inside old result trees.
- **`deeplob_results_fast/`** (nested plots + ad‑hoc `.pt`; canonical weights live in **`Output_models/`**).
- Duplicate notebook **`DeepLOB_training_stablized_final(1).ipynb`** (use **`flow_process/`** copies).
- One‑off **`vol_*.png`** figures.
- Large regenerated **`test.csv` / `test.parquet` / `test_multik.parquet` / `test-20.parquet` / `test_Xy.parquet`** inside `DeepLOB_data_process/` — regenerate with the scripts above; canonical evaluation parquets stay at **repo root** (git‑ignored if large).

See **`.gitignore`** for patterns that keep clones light.
