import argparse
import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path(__file__).parent / "raw_data"
OUTPUT_CSV = Path(__file__).parent / "test.csv"
OUTPUT_PARQUET = Path(__file__).parent / "test.parquet"

TIME_COLS = ["UpdateTime", "TradingDay"]

# LOB 列顺序：五档 Bid（远→近）价量成对，再五档 Ask（远→近），与里程碑 / DeepLOB 论文表述一致
_LEVELS = [5, 4, 3, 2, 1]
LOB_FEATURE_COLS = []
for lv in _LEVELS:
    LOB_FEATURE_COLS.extend([f"BidPrice{lv}", f"BidVolume{lv}"])
for lv in _LEVELS:
    LOB_FEATURE_COLS.extend([f"AskPrice{lv}", f"AskVolume{lv}"])

BID_PRICE_COLS = [f"BidPrice{lv}" for lv in _LEVELS]
ASK_PRICE_COLS = [f"AskPrice{lv}" for lv in _LEVELS]
BID_VOL_COLS = [f"BidVolume{lv}" for lv in _LEVELS]
ASK_VOL_COLS = [f"AskVolume{lv}" for lv in _LEVELS]
_ALL_LOB = LOB_FEATURE_COLS  # 与上面展开一致


def load_and_merge_first_n_csv(n=None):
    """
    加载 raw_data 下 CSV（按文件名排序），纵向 concat。
    n=None 表示全部文件；n 为整数时只取前 n 个。
    """
    csv_files = sorted(RAW_DIR.glob("*.csv"))
    if n is not None:
        csv_files = csv_files[:n]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files in {RAW_DIR}")

    dfs = []
    needed = TIME_COLS + _ALL_LOB
    for f in csv_files:
        df = pd.read_csv(f)
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {f.name}: {missing}")
        df = df[needed].copy()
        df["mid_price"] = (df["BidPrice1"] + df["AskPrice1"]) / 2
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    print(f"Merged {len(csv_files)} files -> shape={merged.shape}")
    return merged


def _forward_mid_ret(mid: np.ndarray, k: int) -> np.ndarray:
    """单日内：mid_ret[i] = (mid[i+k]-mid[i])/mid[i]*1e4（bp）"""
    n = len(mid)
    out = np.full(n, np.nan)
    if k <= 0 or n <= k:
        return out
    out[: n - k] = (mid[k:] - mid[:-k]) / (mid[:-k] + 1e-12) * 10000
    return out


def _doc_smooth_rt(mid: np.ndarray, k: int) -> np.ndarray:
    """
    文档定义：m_+(t)=(1/k)Σ_{i=1..k} p_{t+i}；m_-(t)=(1/k)Σ_{i=0..k-1} p_{t-i}（过去 k 档含当前，与 m_+ 对称）。
    r_t = (m_+ - m_-) / m_-。输出为 r_t×1e4（bp）。有效下标：k-1 <= i <= n-k-1，故单日至少需 2k 个 tick。
    """
    n = len(mid)
    out = np.full(n, np.nan)
    if k <= 0 or n < 2 * k:
        return out
    m = mid.astype(np.float64)
    c = np.r_[0.0, np.cumsum(m)]
    i_vec = np.arange(k - 1, n - k, dtype=np.int64)
    m_minus = (c[i_vec + 1] - c[i_vec - k + 1]) / k
    m_plus = (c[i_vec + k + 1] - c[i_vec + 1]) / k
    ok = m_minus > 0
    out[i_vec[ok]] = (m_plus[ok] - m_minus[ok]) / (m_minus[ok] + 1e-12) * 10000
    return out


def time_bin_30m(series_dt: pd.Series) -> np.ndarray:
    """日内 30 分钟桶索引 0..47"""
    m = series_dt.dt.hour * 60 + series_dt.dt.minute
    return (m // 30).astype(np.int64)


def compute_daily_bin_std_of_r(df: pd.DataFrame, r_col: str = "smooth_ret") -> pd.DataFrame:
    """
    每个交易日、每个 30 分钟桶内，对平滑收益 r_t（smooth_ret）求 std，用于 σ_b 估计。
    返回长表：TradingDay, time_bin, bin_std
    """
    rows = []
    for day, g in df.groupby("TradingDay", sort=True):
        g = g.sort_values("UpdateTime")
        sub = g.assign(time_bin=time_bin_30m(pd.to_datetime(g["UpdateTime"])))
        for b, part in sub.groupby("time_bin"):
            s = part[r_col].std()
            rows.append({"TradingDay": day, "time_bin": int(b), "bin_std": s})
    return pd.DataFrame(rows)


def map_rolling_sigma_bin(std_long: pd.DataFrame, W: int = 5) -> pd.DataFrame:
    """
    对每个 (交易日, time_bin)，sigma = 前 W 个交易日同 bin 的 bin_std 的均值（不含当日）。
    返回列：TradingDay, time_bin, sigma_bin
    """
    if std_long.empty:
        return pd.DataFrame(columns=["TradingDay", "time_bin", "sigma_bin"])
    days = sorted(std_long["TradingDay"].unique())
    bins = sorted(std_long["time_bin"].unique())
    std_mat = pd.DataFrame(index=days, columns=bins, dtype=float)
    for _, row in std_long.iterrows():
        std_mat.loc[row["TradingDay"], row["time_bin"]] = row["bin_std"]
    # 前 W 日均值，不含当日
    rolled = std_mat.shift(1).rolling(window=W, min_periods=1).mean()
    out = rolled.stack().reset_index()
    out.columns = ["TradingDay", "time_bin", "sigma_bin"]
    return out


def _label_col_names(suffix: str):
    """suffix 为 '' 时列名与旧版一致；为 '_k10' 等为多 horizon 导出。"""
    def c(base: str) -> str:
        return f"{base}{suffix}" if suffix else base

    return {
        "mid_ret": c("mid_ret"),
        "smooth_ret": c("smooth_ret"),
        "sigma_bin": c("sigma_bin"),
        "sigma_bin_filled": c("sigma_bin_filled"),
        "vol_weighted_ret": c("vol_weighted_ret"),
        "label": c("label"),
    }


def compute_labels_milestone(
    df: pd.DataFrame,
    horizon_k: int = 20,
    vol_window_days: int = 5,
    threshold_norm: float = 1.0,
    suffix: str = "",
):
    """
    逐日计算 mid_ret、smooth_ret（文档 r_t）；
    σ_b 为各 30 分钟桶内 r_t 的日度 std，再对前 vol_window_days 日同桶取均值；
    vol_weighted_ret = r_t / σ；label 对 vol_weighted_ret 用 threshold_norm（α）三分类。
    suffix：例如 '_k10'，则列名为 mid_ret_k10、label_k10 等；'' 时保持 mid_ret、label（兼容原脚本）。
    """
    df = df.copy()
    df["UpdateTime"] = pd.to_datetime(df["UpdateTime"])
    cn = _label_col_names(suffix)
    smooth_key = cn["smooth_ret"]
    sig_key = cn["sigma_bin"]
    filled_key = cn["sigma_bin_filled"]
    vwr_key = cn["vol_weighted_ret"]
    lab_key = cn["label"]

    mid_ret = np.full(len(df), np.nan)
    smooth_ret = np.full(len(df), np.nan)
    for _, g in df.groupby("TradingDay", sort=False):
        idx = g.index.values
        order = np.argsort(df.loc[idx, "UpdateTime"].values)
        idx_s = idx[order]
        mid = df.loc[idx_s, "mid_price"].values.astype(np.float64)
        mid_ret[idx_s] = _forward_mid_ret(mid, horizon_k)
        smooth_ret[idx_s] = _doc_smooth_rt(mid, horizon_k)

    df[cn["mid_ret"]] = mid_ret
    df[smooth_key] = smooth_ret

    std_long = compute_daily_bin_std_of_r(df, r_col=smooth_key)
    sigma_map = map_rolling_sigma_bin(std_long, W=vol_window_days)
    sigma_map = sigma_map.rename(columns={"sigma_bin": sig_key})
    if "time_bin" not in df.columns:
        df["time_bin"] = time_bin_30m(df["UpdateTime"])
    df = df.merge(sigma_map, on=["TradingDay", "time_bin"], how="left")

    sig = df[sig_key].replace(0, np.nan).fillna(np.nanmedian(df[sig_key]))
    df[filled_key] = sig
    df[vwr_key] = df[smooth_key] / (sig + 1e-12)

    def classify(z):
        if pd.isna(z):
            return np.nan
        if z >= threshold_norm:
            return 1
        if z <= -threshold_norm:
            return -1
        return 0

    df[lab_key] = df[vwr_key].apply(classify)
    return df


def remove_edge_ticks(df, n_remove=10):
    df = df.reset_index(drop=True)
    keep_mask = np.ones(len(df), dtype=bool)
    for _, g in df.groupby("TradingDay"):
        idx = g.index.values
        n = len(idx)
        if n <= 2 * n_remove:
            keep_mask[idx] = False
            continue
        keep_mask[idx[:n_remove]] = False
        keep_mask[idx[-n_remove:]] = False
    return df.loc[keep_mask].reset_index(drop=True)


def output_column_order():
    """时间 + 20 维 LOB + 中间变量与标签（供检查与训练脚本读列名）"""
    extra = [
        "mid_price",
        "mid_ret",
        "smooth_ret",
        "time_bin",
        "sigma_bin",
        "sigma_bin_filled",
        "vol_weighted_ret",
        "label",
    ]
    return TIME_COLS + LOB_FEATURE_COLS + extra


def save_table(df: pd.DataFrame, path_csv: Path, path_parquet: Path):
    df.to_csv(path_csv, index=False)
    print(f"Saved CSV: {path_csv} ({df.shape[0]} rows)")
    try:
        df.to_parquet(path_parquet, index=False)
        print(f"Saved Parquet: {path_parquet}")
    except Exception as e:
        print(f"Parquet skipped ({e}); install pyarrow for fast compressed storage: pip install pyarrow")


def main():
    parser = argparse.ArgumentParser(description="Prepare DeepLOB-style LOB + labels")
    parser.add_argument("--horizon", type=int, default=20, help="预测 horizon k（tick）")
    parser.add_argument("--csv", type=Path, default=OUTPUT_CSV, help="输出 CSV 路径")
    parser.add_argument("--parquet", type=Path, default=OUTPUT_PARQUET, help="输出 Parquet 路径")
    args = parser.parse_args()

    horizon_k = args.horizon
    vol_W = 5
    threshold_norm = 1.0

    print(f"horizon_k={horizon_k}, out_parquet={args.parquet}")

    df = load_and_merge_first_n_csv(n=None)

    df = compute_labels_milestone(
        df, horizon_k=horizon_k, vol_window_days=vol_W, threshold_norm=threshold_norm
    )
    df = df.dropna(subset=["mid_ret", "smooth_ret", "vol_weighted_ret", "label"])

    valid = df["label"]
    total = len(valid)
    print("\n三分类比例（去边缘前，有效 label 行）:")
    for name, v in [("1 (涨)", 1), ("0 (平)", 0), ("-1 (跌)", -1)]:
        c = (valid == v).sum()
        print(f"  {name}: {c:>8} ({100 * c / total:.2f}%)")
    print(f"  Total: {total:>8}")

    df = remove_edge_ticks(df, n_remove=10)

    cols = [c for c in output_column_order() if c in df.columns]
    df = df[cols]

    save_table(df, args.csv, args.parquet)
    print(f"Final shape={df.shape}")


if __name__ == "__main__":
    main()
