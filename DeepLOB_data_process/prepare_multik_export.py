"""
全量 raw_data → 导出：
  - 时间：row_id, UpdateTime, TradingDay
  - X：真实五档订单簿，共 20 列（BidPrice5,BidVolume5,…,AskPrice1,AskVolume1，与 prepare_deeplob 一致）
  - label：k=10/20/50 三分类 + 各自三分类前的连续值 vol_weighted_ret

用法：
  python prepare_multik_export.py
  python prepare_multik_export.py --output test_multik.parquet
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_deeplob_data import (
    LOB_FEATURE_COLS,
    TIME_COLS,
    compute_labels_milestone,
    load_and_merge_first_n_csv,
    remove_edge_ticks,
)

PROJECT = Path(__file__).parent

LABEL_DISCRETE = ["label_k10", "label_k20", "label_k50"]
LABEL_CONT = ["vol_weighted_ret_k10", "vol_weighted_ret_k20", "vol_weighted_ret_k50"]


def build_export_frame(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    out = {
        "row_id": np.arange(n, dtype=np.int64),
        "UpdateTime": df["UpdateTime"].values,
        "TradingDay": df["TradingDay"].values,
    }
    for c in LOB_FEATURE_COLS:
        out[c] = df[c].values
    for c in LABEL_DISCRETE + LABEL_CONT:
        out[c] = df[c].values

    order = ["row_id"] + TIME_COLS + LOB_FEATURE_COLS + LABEL_DISCRETE + LABEL_CONT
    return pd.DataFrame(out)[order]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PROJECT / "test_multik.parquet")
    parser.add_argument("--csv", type=Path, default=None, help="可选，同时写 CSV（大文件慎用）")
    parser.add_argument("--no-edge-trim", action="store_true", help="不做日内首尾各 10 tick 删除")
    args = parser.parse_args()

    vol_W = 5
    threshold_norm = 1.0

    print("Loading and merging all CSV...")
    df = load_and_merge_first_n_csv(n=None)

    for k in (10, 20, 50):
        suf = f"_k{k}"
        print(f"Labels horizon k={k} ...")
        df = compute_labels_milestone(
            df,
            horizon_k=k,
            vol_window_days=vol_W,
            threshold_norm=threshold_norm,
            suffix=suf,
        )

    drop_subset = LABEL_DISCRETE + LABEL_CONT
    df = df.dropna(subset=drop_subset)
    print(f"After dropna (all k valid): {len(df):,} rows")

    if not args.no_edge_trim:
        df = remove_edge_ticks(df, n_remove=10)
        print(f"After remove_edge_ticks: {len(df):,} rows")

    spill = ["time_bin", "mid_price"]
    for k in (10, 20, 50):
        spill.extend(
            [
                f"mid_ret_k{k}",
                f"smooth_ret_k{k}",
                f"sigma_bin_k{k}",
                f"sigma_bin_filled_k{k}",
            ]
        )
    df = df.drop(columns=[c for c in spill if c in df.columns], errors="ignore")

    df = df.reset_index(drop=True)
    out_df = build_export_frame(df)

    out_df.to_parquet(args.output, index=False)
    print(f"Saved parquet: {args.output}  shape={out_df.shape}")

    if args.csv is not None:
        out_df.to_csv(args.csv, index=False)
        print(f"Saved csv: {args.csv}")


if __name__ == "__main__":
    main()
