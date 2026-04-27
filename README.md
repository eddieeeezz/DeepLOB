# DeepLOB

This project implements and evaluates a stabilized DeepLOB-style neural network for limit order book prediction. It includes training notebooks for three datasets and a separate evaluation notebook that reloads saved checkpoints and reproduces test-set metrics.

## Project Structure

```text
DeepLOB/
|-- DeepLOB_training_stablized_final.ipynb
|-- DeepLOB_training_stablized_final_k10.ipynb
|-- DeepLOB_training_stablized_final_k50.ipynb
|-- DeepLOB_evaluate_trained_models.ipynb
|-- deeplob_eval/
|   |-- __init__.py
|   |-- data.py
|   |-- evaluation.py
|   `-- model.py
|-- Output_models/
|   |-- deeplob5_stabilized_best.pth
|   |-- deeplob5_stabilized_best_10.pth
|   `-- deeplob5_stabilized_best_50.pth
|-- test_Xy.parquet
|-- test_10.parquet
`-- test_50.parquet
```

## Notebooks

- `DeepLOB_training_stablized_final.ipynb`: training workflow for the base dataset.
- `DeepLOB_training_stablized_final_k10.ipynb`: training workflow using `test_10.parquet`.
- `DeepLOB_training_stablized_final_k50.ipynb`: training workflow using `test_50.parquet`.
- `DeepLOB_evaluate_trained_models.ipynb`: loads the three trained checkpoints and evaluates each model on its matching test split.

The evaluation notebook is split into three sections:

- k10 model on `test_10.parquet`
- k50 model on `test_50.parquet`
- base model on `test_Xy.parquet`

Each section reports test metrics, a confusion matrix, and per-class precision, recall, and F1.

## Model Checkpoints

The saved models are stored in `Output_models/`:

| Dataset | Checkpoint |
|---|---|
| `test_Xy.parquet` | `Output_models/deeplob5_stabilized_best.pth` |
| `test_10.parquet` | `Output_models/deeplob5_stabilized_best_10.pth` |
| `test_50.parquet` | `Output_models/deeplob5_stabilized_best_50.pth` |

The checkpoints contain the model state dict and metadata such as feature columns, sequence length, hidden size, batch size, best epoch, and best validation macro F1.

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

Note: the parquet datasets are large. GitHub does not allow files larger than 100 MB in a normal repository. If these files are not included in the repository, place them in the project root before running the notebooks:

```text
test_Xy.parquet
test_10.parquet
test_50.parquet
```

## Local Package

Reusable model and evaluation code is stored in `deeplob_eval/`. The evaluation notebook imports it like a local package:

```python
from deeplob_eval import DeepLOB5Stable, run_checkpoint_evaluation
```

This avoids hard-coded system paths. The project can be moved to another folder as long as the notebook is run from the project root.

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

2. Make sure the parquet data files are in the project root.

3. Make sure the trained checkpoints are in `Output_models/`.

4. Start Jupyter:

```bash
jupyter notebook
```

5. Open and run:

```text
DeepLOB_evaluate_trained_models.ipynb
```

To retrain models, run the corresponding training notebook first, then rerun the evaluation notebook.

## Evaluation Outputs

For each trained model, the evaluation notebook reports:

- best epoch loaded from the checkpoint
- best validation macro F1 loaded from the checkpoint
- test loss
- test accuracy
- test macro F1
- confusion matrix
- per-class precision, recall, F1, and support

## Reference

This project is based on the DeepLOB architecture for limit order book prediction:

Zhang, Z., Zohren, S., & Roberts, S. (2019). DeepLOB: Deep Convolutional Neural Networks for Limit Order Books.
