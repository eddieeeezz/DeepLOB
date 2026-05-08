"""
从全量 test.parquet（prepare_deeplob_data.py 默认 k=20、合并全部 CSV）导出：
仅 index + 20 维 LOB X + label → 默认保存为 test-20.parquet。

使用前请先跑全量：python prepare_deeplob_data.py
其他 horizon 可先：python prepare_deeplob_data.py --horizon 10 --parquet test_10.parquet
再：python export_test_xy.py --input test_10.parquet --output test-10.parquet
"""

import argparse
from pathlib import Path

import pandas as pd

_LEVELS = [5, 4, 3, 2, 1]
LOB_X_COLS = []
for lv in _LEVELS:
    LOB_X_COLS.extend([f"BidPrice{lv}", f"BidVolume{lv}"])
for lv in _LEVELS:
    LOB_X_COLS.extend([f"AskPrice{lv}", f"AskVolume{lv}"])

PROJECT = Path(__file__).parent


def main():
    p = argparse.ArgumentParser(
        description="Export index + 20 LOB + label (k=20 全量对应默认 test.parquet → test-20.parquet)"
    )
    p.add_argument(
        "--input",
        type=Path,
        default=PROJECT / "test.parquet",
        help="全量特征+标签 parquet（默认 test.parquet，horizon 需与文件名约定一致）",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "test-20.parquet",
        help="仅 X+label 输出路径",
    )
    args = p.parse_args()

    df = pd.read_parquet(args.input)
    df = df.reset_index(drop=True)

    missing = [c for c in LOB_X_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"输入缺少 X 列 ({len(missing)} 个): {missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    if "label" not in df.columns:
        raise ValueError("输入缺少列 label")

    out = pd.DataFrame({"index": df.index.to_numpy()})
    for c in LOB_X_COLS:
        out[c] = df[c].values
    out["label"] = df["label"].values

    out.to_parquet(args.output, index=False)
    print(f"Saved {args.output}")
    print(f"shape: {out.shape}  (columns: index + {len(LOB_X_COLS)} X + label)")


if __name__ == "__main__":
    main()
