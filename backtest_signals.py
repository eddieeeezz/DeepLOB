"""
Backtest LOB strategy from model predictions (y_pred_label: +1 / 0 / -1).

Rules (per user spec):
- +1: long — buy at Ask1, schedule cover at Bid1 after HOLD_TICKS; another +1 before exit defers to new+20;
        -1 before exit closes long immediately and flips short.
- -1: short — sell at Bid1, schedule cover at Ask1 after HOLD_TICKS; another -1 before exit defers;
        +1 before exit closes short immediately and flips long.

At each tick: (1) execute scheduled exit if due; (2) apply discrete signal if present.
Signals exist only on rows present in the predictions file (window_end_iloc).

Requires the same parquet as training (BidPrice1, AskPrice1, index) and predictions parquet from generate_predictions.py.

By default prints summary metrics only; use --equity-csv / --trades-csv / --plot to write optional artifacts.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Match generate_predictions.py defaults
TRAIN_FRAC = 0.70
TEST_FRAC = 0.02
HOLD_TICKS_DEFAULT = 20


@dataclass
class Trade:
    side: str  # "long" | "short"
    entry_iloc: int
    exit_iloc: int
    entry_price: float
    exit_price: float
    entry_mid: float
    exit_mid: float
    pnl: float  # spread-aware execution PnL
    pnl_mid: float  # mid-to-mid PnL (ignores spread)


@dataclass
class BacktestState:
    pos: int = 0  # 0 flat, 1 long, -1 short
    entry_price: float = np.nan
    entry_mid: float = np.nan
    entry_iloc: int = -1
    exit_iloc: int | None = None
    realized: float = 0.0
    trades: list[Trade] = field(default_factory=list)

    def close_long(self, iloc: int, bid: float, ask: float) -> None:
        pnl = float(bid - self.entry_price)
        exit_mid = float((bid + ask) / 2.0)
        pnl_mid = float(exit_mid - self.entry_mid)
        self.realized += pnl
        self.trades.append(
            Trade(
                "long",
                self.entry_iloc,
                iloc,
                self.entry_price,
                bid,
                self.entry_mid,
                exit_mid,
                pnl,
                pnl_mid,
            )
        )
        self.pos = 0
        self.entry_price = np.nan
        self.entry_mid = np.nan
        self.entry_iloc = -1
        self.exit_iloc = None

    def close_short(self, iloc: int, ask: float, bid: float) -> None:
        pnl = float(self.entry_price - ask)
        exit_mid = float((bid + ask) / 2.0)
        pnl_mid = float(self.entry_mid - exit_mid)
        self.realized += pnl
        self.trades.append(
            Trade(
                "short",
                self.entry_iloc,
                iloc,
                self.entry_price,
                ask,
                self.entry_mid,
                exit_mid,
                pnl,
                pnl_mid,
            )
        )
        self.pos = 0
        self.entry_price = np.nan
        self.entry_mid = np.nan
        self.entry_iloc = -1
        self.exit_iloc = None


def _clip_exit(idx: int, hold: int, n: int) -> int:
    return int(min(idx + hold, n - 1))


def _is_sig(s: float, target: int) -> bool:
    return abs(float(s) - float(target)) < 1e-6


def run_backtest(
    bid: np.ndarray,
    ask: np.ndarray,
    valid_end: int,
    signals: dict[int, float],
    hold_ticks: int,
) -> tuple[np.ndarray, np.ndarray, BacktestState]:
    n = len(bid)
    st = BacktestState()
    ilocs = np.arange(valid_end, n, dtype=np.int64)
    equity = np.full(len(ilocs), np.nan, dtype=np.float64)

    for j, idx in enumerate(ilocs):
        b, a = float(bid[idx]), float(ask[idx])
        if not np.isfinite(b) or not np.isfinite(a):
            unreal = 0.0
            if st.pos == 1 and np.isfinite(st.entry_price) and np.isfinite(b):
                unreal = b - st.entry_price
            elif st.pos == -1 and np.isfinite(st.entry_price) and np.isfinite(a):
                unreal = st.entry_price - a
            equity[j] = st.realized + unreal
            continue

        # (1) Scheduled exit
        if st.exit_iloc is not None and idx == st.exit_iloc and st.pos != 0:
            if st.pos == 1:
                st.close_long(idx, b, a)
            else:
                st.close_short(idx, a, b)

        # (2) Discrete signal
        if idx in signals:
            s = signals[idx]
            if _is_sig(s, 0):
                pass
            elif _is_sig(s, 1):
                if st.pos == 0:
                    st.pos = 1
                    st.entry_price = a
                    st.entry_mid = float((a + b) / 2.0)
                    st.entry_iloc = int(idx)
                    st.exit_iloc = _clip_exit(idx, hold_ticks, n)
                elif st.pos == 1:
                    st.exit_iloc = _clip_exit(idx, hold_ticks, n)
                else:  # short -> flip long
                    st.close_short(idx, a, b)
                    st.pos = 1
                    st.entry_price = a
                    st.entry_mid = float((a + b) / 2.0)
                    st.entry_iloc = int(idx)
                    st.exit_iloc = _clip_exit(idx, hold_ticks, n)
            elif _is_sig(s, -1):
                if st.pos == 0:
                    st.pos = -1
                    st.entry_price = b
                    st.entry_mid = float((a + b) / 2.0)
                    st.entry_iloc = int(idx)
                    st.exit_iloc = _clip_exit(idx, hold_ticks, n)
                elif st.pos == 1:
                    st.close_long(idx, b, a)
                    st.pos = -1
                    st.entry_price = b
                    st.entry_mid = float((a + b) / 2.0)
                    st.entry_iloc = int(idx)
                    st.exit_iloc = _clip_exit(idx, hold_ticks, n)
                else:
                    st.exit_iloc = _clip_exit(idx, hold_ticks, n)

        unreal = 0.0
        if st.pos == 1 and np.isfinite(st.entry_price):
            unreal = b - st.entry_price
        elif st.pos == -1 and np.isfinite(st.entry_price):
            unreal = st.entry_price - a
        equity[j] = st.realized + unreal

    # Force flat at last tick if still open
    last = n - 1
    if st.pos == 1 and np.isfinite(bid[last]):
        st.close_long(last, float(bid[last]), float(ask[last]))
    elif st.pos == -1 and np.isfinite(ask[last]):
        st.close_short(last, float(ask[last]), float(bid[last]))
    if len(equity):
        equity[-1] = st.realized

    return ilocs, equity, st


def max_drawdown(equity: np.ndarray) -> float:
    x = equity[np.isfinite(equity)]
    if x.size == 0:
        return 0.0
    peak = np.maximum.accumulate(x)
    dd = (peak - x) / np.where(peak > 0, peak, 1.0)
    return float(np.max(dd)) if dd.size else 0.0


def average_spread_bps(bid: np.ndarray, ask: np.ndarray) -> float:
    mid = (bid + ask) / 2.0
    mask = np.isfinite(bid) & np.isfinite(ask) & np.isfinite(mid) & (mid > 0)
    if not np.any(mask):
        return 0.0
    spread_bps = (ask[mask] - bid[mask]) / mid[mask] * 10000.0
    return float(np.mean(spread_bps))


def summarize(st: BacktestState, equity: np.ndarray, ilocs: np.ndarray) -> dict:
    trades = st.trades
    wins_spread = sum(1 for t in trades if t.pnl > 0)
    losses_spread = sum(1 for t in trades if t.pnl < 0)
    flat_spread = sum(1 for t in trades if t.pnl == 0)
    wins_mid = sum(1 for t in trades if t.pnl_mid > 0)
    losses_mid = sum(1 for t in trades if t.pnl_mid < 0)
    flat_mid = sum(1 for t in trades if t.pnl_mid == 0)
    rets = np.diff(equity[np.isfinite(equity)])
    rets = rets[np.isfinite(rets)]
    sharpe = 0.0
    if rets.size > 1 and np.std(rets) > 1e-12:
        sharpe = float(np.mean(rets) / np.std(rets) * np.sqrt(len(rets)))

    return {
        "n_trades": len(trades),
        "winning_trades_spread": wins_spread,
        "losing_trades_spread": losses_spread,
        "breakeven_trades_spread": flat_spread,
        "winning_trades_mid": wins_mid,
        "losing_trades_mid": losses_mid,
        "breakeven_trades_mid": flat_mid,
        "win_rate_spread": wins_spread / len(trades) if trades else 0.0,
        "win_rate_mid": wins_mid / len(trades) if trades else 0.0,
        "total_pnl": st.realized,
        "final_equity": float(equity[np.isfinite(equity)][-1]) if np.any(np.isfinite(equity)) else 0.0,
        "max_drawdown": max_drawdown(equity),
        "sharpe_like": sharpe,
        "mean_trade_pnl_spread": float(np.mean([t.pnl for t in trades])) if trades else 0.0,
        "mean_trade_pnl_mid": float(np.mean([t.pnl_mid for t in trades])) if trades else 0.0,
    }


def maybe_plot(ilocs: np.ndarray, equity: np.ndarray, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skip plot.")
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ilocs, equity, lw=0.8, color="#1f77b4")
    ax.set_title("Equity (realized + mark-to-market)")
    ax.set_xlabel("row iloc")
    ax.set_ylabel("cumulative PnL (price units)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print("Saved plot:", out_path.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest +1/-1 prediction signals on LOB.")
    root = Path(__file__).resolve().parent
    parser.add_argument(
        "--data",
        type=Path,
        default=root / "test_Xy.parquet",
        help="LOB parquet (needs index, BidPrice1, AskPrice1)",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=root / "predictions_deeplob5_best_last2pct.parquet",
        help="Output from generate_predictions.py",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=TEST_FRAC,
        help="Tail fraction; must match generate_predictions TEST_FRAC (default 0.02)",
    )
    parser.add_argument("--hold-ticks", type=int, default=HOLD_TICKS_DEFAULT)
    parser.add_argument(
        "--equity-csv",
        type=Path,
        default=None,
        help="If set, write equity time series CSV to this path",
    )
    parser.add_argument(
        "--trades-csv",
        type=Path,
        default=None,
        help="If set, write closed trades CSV to this path",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=None,
        help="If set, save equity curve PNG (needs matplotlib)",
    )
    args = parser.parse_args()

    if not args.data.is_file():
        raise SystemExit(f"Data not found: {args.data}")
    if not args.predictions.is_file():
        raise SystemExit(f"Predictions not found: {args.predictions}")

    cols = ["index", "BidPrice1", "AskPrice1"]
    df = pd.read_parquet(args.data, columns=cols)
    df = df.sort_values("index").reset_index(drop=True)
    n = len(df)
    valid_end = int(n * (1.0 - args.test_frac))
    bid = df["BidPrice1"].to_numpy(dtype=np.float64)
    ask = df["AskPrice1"].to_numpy(dtype=np.float64)
    avg_spread_bps = average_spread_bps(bid[valid_end:], ask[valid_end:])

    pred = pd.read_parquet(args.predictions)
    if "window_end_iloc" not in pred.columns or "y_pred_label" not in pred.columns:
        raise SystemExit("predictions parquet must contain window_end_iloc, y_pred_label")

    # last wins if duplicate iloc
    signals = {}
    for iloc, lab in zip(
        pred["window_end_iloc"].astype(np.int64),
        pred["y_pred_label"].astype(np.float64),
    ):
        signals[int(iloc)] = float(lab)

    ilocs, equity, st = run_backtest(
        bid, ask, valid_end, signals, args.hold_ticks
    )

    if args.equity_csv is not None:
        curve = pd.DataFrame({"iloc": ilocs, "equity": equity})
        args.equity_csv.parent.mkdir(parents=True, exist_ok=True)
        curve.to_csv(args.equity_csv, index=False)

    stats = summarize(st, equity, ilocs)
    print("--- Backtest summary ---")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"  avg_spread_bps_test_segment: {avg_spread_bps}")
    if args.equity_csv is not None:
        print("  equity_csv:", args.equity_csv.resolve())
    if args.plot is not None:
        maybe_plot(ilocs, equity, args.plot)

    if args.trades_csv is not None and st.trades:
        args.trades_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([t.__dict__ for t in st.trades]).to_csv(args.trades_csv, index=False)
        print("  trades_csv:", args.trades_csv.resolve())


if __name__ == "__main__":
    main()
