# DeepLOB

This project implements and evaluates a stabilized DeepLOB-style neural network for limit order book prediction on 5-level futures limit order book snapshots. It includes data preparation scripts, training notebooks, trained checkpoints, inference utilities, and a portfolio-style backtest notebook that converts model probabilities into notional target positions.

**Architecture & data flow (diagram + script map):** **[ARCHITECTURE.md](ARCHITECTURE.md)**  

## Project Structure

```text
DeepLOB/
|-- ARCHITECTURE.md              # pipeline diagram + responsibilities
|-- README.md                    # this file
|-- .gitignore                   # ignores large regenerated artifacts
|-- flow_process/                # canonical training notebooks (base / k10 / k50)
|-- DeepLOB_data_process/        # ETL only: raw_data + prepare_* + export_*
|-- DeepLOB_evaluate_trained_models.ipynb
|-- generate_predictions.py      # inference → predictions parquet
|-- backtest_signals.py          # simple fixed-unit signal backtest
|-- DeepLOB_probability_portfolio_backtest.ipynb
|-- Output_models/               # best .pth checkpoints
`-- test_Xy.parquet              # base evaluation/backtest dataset
```

## Data processing (`DeepLOB_data_process/`)

LOB snapshots live in **`DeepLOB_data_process/raw_data/`**. Scripts merge raw CSVs, compute smoothed and volatility-normalized labels, and export slim parquet files with `index + 20 LOB features + label`.

The label is smoothed, but the LOB feature columns are not smoothed prices. Backtests use raw `BidPrice1` / `AskPrice1` for execution.

## Notebooks

- `flow_process/DeepLOB_training_stablized_final.ipynb`: training workflow for the base dataset.
- `flow_process/DeepLOB_training_stablized_final_k10.ipynb`: training workflow using `test_10.parquet`.
- `flow_process/DeepLOB_training_stablized_final_k50.ipynb`: training workflow using `test_50.parquet`.
- `DeepLOB_probability_portfolio_backtest.ipynb`: loads a trained checkpoint, generates model probabilities on the held-out test segment, converts `P(up) - P(down)` into probability-sized target notional, and runs a Bid/Ask execution portfolio backtest.
- `DeepLOB_evaluate_trained_models.ipynb`: legacy evaluation notebook for the three saved checkpoints. It expects a local `deeplob_eval` helper package; if that package is not present, use the standalone model definition in the portfolio backtest notebook or refactor the evaluation notebook accordingly.

The portfolio backtest notebook uses:

- initial capital: `1,000,000`
- raw Bid/Ask execution prices
- `EXECUTION_LAG_TICKS = 1`
- trailing probability smoothing, avoiding lookahead bias
- `REBALANCE_INTERVAL_TICKS = 20`, aligned with the base k=20 smoothed label horizon
- target notional capped by a fraction of current NAV

## Model Checkpoints

The saved models are stored in `Output_models/`:

| Dataset | Checkpoint |
|---|---|
| `test_Xy.parquet` | `Output_models/deeplob5_stabilized_best.pth` |
| `test_10.parquet` | `Output_models/deeplob5_stabilized_best_10.pth` |
| `test_50.parquet` | `Output_models/deeplob5_stabilized_best_50.pth` |

The checkpoints contain the model state dict and metadata such as feature columns, sequence length, hidden size, batch size, best epoch, and best validation macro F1. The current portfolio notebook defaults to the base k=20 checkpoint and `test_Xy.parquet`.

## Data

The model uses 20 limit order book features:

- Bid prices and volumes from levels 1 through 5
- Ask prices and volumes from levels 1 through 5
- A three-class label mapped as:
  - `-1.0 -> 0` (`down`)
  - `0.0 -> 1` (`stationary`)
  - `1.0 -> 2` (`up`)

The training and evaluation notebooks use the same chronological split:

- 70% train
- 15% validation
- 15% test

The train split mean and standard deviation are used to normalize validation and test data.

Note: the parquet datasets are large. GitHub does not allow files larger than 100 MB in a normal repository. If these files are not included in the repository, place them in the project root before running the notebooks, or regenerate them under `DeepLOB_data_process/` and copy the exports to the root as needed.

For the default portfolio backtest, `test_Xy.parquet` is required.

## Backtesting Notes

The project contains two backtesting paths:

- `backtest_signals.py`: a simple fixed-unit signal backtest using exported predictions. It is useful as a quick sanity check.
- `DeepLOB_probability_portfolio_backtest.ipynb`: the preferred research notebook for portfolio-style testing. It uses model probabilities, target notional sizing, NAV accounting, turnover, drawdown, and Bid/Ask execution.

Classification metrics do not directly imply trading profitability. The model predicts smoothed future mid-price labels, while executable PnL is measured using raw Bid/Ask prices. Spread and turnover can dominate the signal even when test accuracy or macro F1 looks reasonable.

## Setup

Install the main Python dependencies:

```bash
pip install numpy pandas pyarrow torch matplotlib jupyter
```

If you use a CUDA-enabled GPU, install the PyTorch build that matches your CUDA version from the official PyTorch installation guide.

## Usage

1. Clone the repository:

```bash
git clone https://github.com/eddieeeezz/DeepLOB.git
cd DeepLOB
```

2. Prepare or place the parquet data files in the project root (see **Data processing** above).

3. Make sure the trained checkpoints are in `Output_models/`.

4. Start Jupyter:

```bash
jupyter notebook
```

5. Open and run:

```text
DeepLOB_probability_portfolio_backtest.ipynb
```

To retrain models, run the corresponding training notebook under `flow_process/` first, then update the checkpoint path in the portfolio backtest notebook.

## Portfolio Backtest Outputs

The portfolio backtest notebook reports:

- checkpoint metadata
- test-set label distribution
- model probability outputs
- smoothed target position schedule
- portfolio NAV and return
- max drawdown
- turnover
- gross notional exposure
- trade/rebalance logs

## Reference

This project is based on the DeepLOB architecture for limit order book prediction:

Zhang, Z., Zohren, S., & Roberts, S. (2019). DeepLOB: Deep Convolutional Neural Networks for Limit Order Books.
