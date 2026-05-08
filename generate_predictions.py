"""
Load deeplob5_stabilized_best.pth and export per-window predictions for the last 2% of the timeline
(same layout as the notebook test split, but the tail is 2% instead of 15%).

Z-score uses mean/std from the first 70% only, matching the training notebooks.

By default, directional +1 / -1 are gated with softmax floors (default 0.52) plus a margin between
the top two class probabilities (default 0.06), so near-ties and weak winners map to 0.

Each row is one prediction at the *end* row of a length-`seq_len` window.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# First 70%: training split in notebooks (used only for mu/sigma here).
TRAIN_FRAC = 0.70
# Last 2% of rows: inference region (notebook used last 15%, i.e. valid_end = int(n * 0.85)).
TEST_FRAC = 0.02

DEFAULT_FEATURE_COLS = [
    "BidPrice5",
    "BidVolume5",
    "BidPrice4",
    "BidVolume4",
    "BidPrice3",
    "BidVolume3",
    "BidPrice2",
    "BidVolume2",
    "BidPrice1",
    "BidVolume1",
    "AskPrice5",
    "AskVolume5",
    "AskPrice4",
    "AskVolume4",
    "AskPrice3",
    "AskVolume3",
    "AskPrice2",
    "AskVolume2",
    "AskPrice1",
    "AskVolume1",
]

LABEL_MAP = {-1.0: 0, 0.0: 1, 1.0: 2}
INV_LABEL = {0: -1.0, 1: 0.0, 2: 1.0}


class LOBSequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int):
        assert len(X) == len(y)
        assert len(X) >= seq_len
        self.X = X
        self.y = y
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.X) - self.seq_len + 1

    def __getitem__(self, idx: int):
        x_seq = self.X[idx : idx + self.seq_len]
        y_target = self.y[idx + self.seq_len - 1]
        return torch.tensor(x_seq, dtype=torch.float32), torch.tensor(y_target, dtype=torch.long)


class DeepLOB5Stable(nn.Module):
    def __init__(self, num_classes: int = 3, hidden_size: int = 64):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 2), padding=(0, 1)),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(16, 16, kernel_size=(3, 1), padding=(1, 0)),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(16, 32, kernel_size=(1, 2), padding=(0, 0)),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 1), padding=0),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 1), padding=0),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(32, 32, kernel_size=(3, 1), padding=(1, 0)),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.branch5 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 1), padding=0),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(32, 32, kernel_size=(5, 1), padding=(2, 0)),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(32, 32, kernel_size=(1, 1), padding=0),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.conv_block(x)
        b1 = self.branch1(x)
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        bp = self.branch_pool(x)
        x = torch.cat([b1, b3, b5, bp], dim=1)
        x = x.mean(dim=3)
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.dropout(x)
        return self.fc(x)


@torch.no_grad()
def collect_logits_and_preds(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    progress_label: str = "Inference",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_logits = []
    all_preds = []
    all_targets = []
    n_batches = len(loader)
    for batch_idx, (xb, yb) in enumerate(loader):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        preds = torch.argmax(logits, dim=1)
        all_logits.append(logits.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_targets.append(yb.cpu().numpy())
        pct = 100.0 * (batch_idx + 1) / n_batches
        print(f"\r{progress_label}: {pct:6.2f}% ({batch_idx + 1}/{n_batches} batches)", end="", flush=True)
    print()
    return (
        np.concatenate(all_logits, axis=0),
        np.concatenate(all_targets, axis=0),
        np.concatenate(all_preds, axis=0),
    )


def iloc_ends_test(valid_end: int, n_total: int, seq_len: int) -> np.ndarray:
    n_t = n_total - valid_end
    return valid_end + np.arange(seq_len - 1, n_t, dtype=np.int64)


def threshold_direction_labels(
    logits: np.ndarray,
    min_up_prob: float,
    min_down_prob: float,
    *,
    min_prob_margin: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reduce +1 / -1 frequency: argmax must be up/down, pass absolute prob floors, and pass
    (P(winner) - P(second place)) >= min_prob_margin so near-ties map to stationary.

    Returns (y_pred_cls_gated, y_pred_cls_argmax, probs) where probs is (N, 3).
    """
    probs = torch.softmax(torch.from_numpy(logits.astype(np.float32)), dim=1).numpy()
    raw = np.argmax(logits, axis=1).astype(np.int64)
    p0, p1, p2 = probs[:, 0], probs[:, 1], probs[:, 2]
    out = np.ones_like(raw, dtype=np.int64)
    margin_up = p2 - np.maximum(p0, p1)
    margin_down = p0 - np.maximum(p1, p2)
    up_ok = (raw == 2) & (p2 >= min_up_prob) & (margin_up >= min_prob_margin)
    down_ok = (raw == 0) & (p0 >= min_down_prob) & (margin_down >= min_prob_margin)
    out[up_ok] = 2
    out[down_ok] = 0
    out[raw == 1] = 1
    return out, raw, probs


def main() -> None:
    parser = argparse.ArgumentParser(description="Export DeepLOB predictions for best checkpoint.")
    root = Path(__file__).resolve().parent
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=root / "Output_models" / "deeplob5_stabilized_best.pth",
        help="Path to .pth (default: Output_models/deeplob5_stabilized_best.pth)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=root / "test_Xy.parquet",
        help="Parquet matching this checkpoint (default: test_Xy.parquet)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "predictions_deeplob5_best_last2pct.parquet",
        help="Output parquet path",
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--device",
        default="auto",
        help="cuda, cpu, or auto",
    )
    parser.add_argument(
        "--min-up-prob",
        type=float,
        default=0.52,
        help="Min softmax P(up) to emit +1 when argmax is up (default: 0.52)",
    )
    parser.add_argument(
        "--min-down-prob",
        type=float,
        default=0.52,
        help="Min softmax P(down) to emit -1 when argmax is down (default: 0.52)",
    )
    parser.add_argument(
        "--min-prob-margin",
        type=float,
        default=0.06,
        help="Require P(winner)-P(second) >= this for up/down (default: 0.06; set 0 to disable margin)",
    )
    parser.add_argument(
        "--argmax-labels",
        action="store_true",
        help="Use plain argmax for y_pred_label (no thresholding; more +1/-1)",
    )
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
    if not args.data.is_file():
        raise SystemExit(f"Data file not found: {args.data}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    feature_cols = ckpt.get("feature_cols") or DEFAULT_FEATURE_COLS
    seq_len = int(ckpt.get("seq_len", 100))
    num_classes = int(ckpt.get("num_classes", 3))
    hidden_size = int(ckpt.get("hidden_size", 64))

    label_col = "label"
    load_cols = ["index"] + list(feature_cols) + [label_col]
    df = pd.read_parquet(args.data, columns=load_cols)
    df = df.sort_values("index").reset_index(drop=True)

    df["label_cls"] = df[label_col].map(LABEL_MAP).astype(np.int64)

    n = len(df)
    train_end = int(n * TRAIN_FRAC)
    valid_end = int(n * (1.0 - TEST_FRAC))

    train_df = df.iloc[:train_end]
    test_df = df.iloc[valid_end:]

    X_train_raw = train_df[feature_cols].to_numpy(dtype=np.float32)
    X_test_raw = test_df[feature_cols].to_numpy(dtype=np.float32)
    y_test = test_df["label_cls"].to_numpy(dtype=np.int64)

    train_mean = X_train_raw.mean(axis=0)
    train_std = X_train_raw.std(axis=0)
    train_std = np.where(train_std < 1e-8, 1.0, train_std)

    X_test = (X_test_raw - train_mean) / train_std
    test_ds = LOBSequenceDataset(X_test, y_test, seq_len)

    pin = torch.cuda.is_available()
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=pin
    )

    model = DeepLOB5Stable(num_classes=num_classes, hidden_size=hidden_size).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(
        f"Test segment (last {100 * TEST_FRAC:.0f}%): rows iloc [{valid_end}, {n}), "
        f"{len(test_ds)} prediction windows"
    )
    logits, y_true, y_pred_argmax = collect_logits_and_preds(
        model, test_loader, device, progress_label="Test split inference"
    )
    if args.argmax_labels:
        y_pred = y_pred_argmax
        probs = torch.softmax(torch.from_numpy(logits.astype(np.float32)), dim=1).numpy()
    else:
        if not (0.0 < args.min_up_prob <= 1.0 and 0.0 < args.min_down_prob <= 1.0):
            raise SystemExit("--min-up-prob and --min-down-prob must be in (0, 1]")
        if args.min_prob_margin < 0 or args.min_prob_margin > 1:
            raise SystemExit("--min-prob-margin must be in [0, 1]")
        y_pred, y_pred_argmax, probs = threshold_direction_labels(
            logits,
            args.min_up_prob,
            args.min_down_prob,
            min_prob_margin=args.min_prob_margin,
        )

    iloc_end = iloc_ends_test(valid_end, n, seq_len)
    if len(iloc_end) != len(y_pred):
        raise RuntimeError(f"Length mismatch: iloc {len(iloc_end)} vs preds {len(y_pred)}")

    data_index = df["index"].to_numpy()[iloc_end]
    rows = [
        {
            "split": "test",
            "window_end_iloc": int(iloc_end[i]),
            "data_index": data_index[i],
            "y_true_cls": int(y_true[i]),
            "y_pred_cls": int(y_pred[i]),
            "y_pred_cls_argmax": int(y_pred_argmax[i]),
            "y_true_label": INV_LABEL[int(y_true[i])],
            "y_pred_label": INV_LABEL[int(y_pred[i])],
            "y_pred_label_argmax": INV_LABEL[int(y_pred_argmax[i])],
            "prob_down": float(probs[i, 0]),
            "prob_stationary": float(probs[i, 1]),
            "prob_up": float(probs[i, 2]),
            "logit_down": float(logits[i, 0]),
            "logit_stationary": float(logits[i, 1]),
            "logit_up": float(logits[i, 2]),
        }
        for i in range(len(y_pred))
    ]

    out_df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.output, index=False)

    print("Device:", device)
    print("Checkpoint:", args.checkpoint)
    print("Data:", args.data)
    print("Rows written:", len(out_df))
    print("Output:", args.output.resolve())
    print("y_pred_label counts (gated):\n", out_df["y_pred_label"].value_counts().sort_index())
    if not args.argmax_labels:
        nchg = int((out_df["y_pred_label"] != out_df["y_pred_label_argmax"]).sum())
        print(
            f"(gates: min_up={args.min_up_prob}, min_down={args.min_down_prob}, "
            f"min_prob_margin={args.min_prob_margin}; rows relabeled vs argmax: {nchg} / {len(out_df)} "
            f"({100.0 * nchg / max(len(out_df), 1):.2f}%))"
        )
        print("y_pred_label_argmax counts:\n", out_df["y_pred_label_argmax"].value_counts().sort_index())
        print(
            "Tip: if +1/-1 still too frequent, raise --min-up-prob/--min-down-prob (e.g. 0.58) "
            "or --min-prob-margin (e.g. 0.10); use --min-prob-margin 0 to only use prob floors."
        )


if __name__ == "__main__":
    main()
