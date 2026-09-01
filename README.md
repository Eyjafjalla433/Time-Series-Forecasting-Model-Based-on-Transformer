# TS Transformer Forecasting

PyTorch-based time series forecasting project with:

- an encoder-decoder Transformer
- data preprocessing for TFB-style CSV files
- offline rolling evaluation
- ARIMA, ETS, RNN, and TCN baselines
- plotting and metrics utilities

## Requirements

- Python 3.9+
- Install dependencies:

```bash
pip install -r requirements.txt
```

If you need a specific CUDA build, install PyTorch from the official site first, then install the remaining packages.

## Project Layout

```text
configs/        experiment config
data/           raw and processed datasets
models/         Transformer model code
engine/         training and inference logic
benchmarks/     ARIMA / ETS / RNN / TCN baselines
scripts/        metrics, visualization, benchmark helpers
run_train.py    training entry
run_infer.py    inference / offline evaluation entry
```

## Data Format

The main pipeline uses numeric wide CSV tables with shape `[T, F]`.

- the first `src_dim` columns are used as model inputs
- the last `tgt_dim` columns are used for decoder targets
- the last `out_dim` columns are used as prediction outputs

Because of this, the target column should usually be moved to the end of the table during preprocessing.

Raw files in `data/dataset/` follow the TFB-style long format with:

- `date`
- `data`
- `cols`

## Preprocessing

Example:

```bash
python data/preprocessing.py ^
  --input-path data/dataset/Weather.csv ^
  --output-path data/processed/Weather_wide.csv ^
  --future-ratio 0.2 ^
  --target-cols OT
```

This script can:

- convert long-format CSV to wide format
- split data into `*_train.csv` and `*_future.csv`
- optionally standardize the output tables

Note: the current Transformer train/infer pipeline does not actively use normalization stats at runtime. If you standardize data, train and infer on the standardized CSV files directly.

## Default Setup

The active config is [`configs/default.yaml`].

Key defaults:

- `src_dim: 21`
- `input_length: 12`
- `pred_length: 6`
- `batch_size: 16`
- `epochs: 50`
- `train_csv: data/processed/Weather_wide_train.csv`
- `future_path: data/processed/Weather_wide_future.csv`

If `cuda` is requested but unavailable, the code falls back to `cpu`.

## Train

```bash
python run_train.py
```

Outputs are controlled by `configs/default.yaml`. By default, training writes:

- checkpoint: `checkpoints/Weather_model.pt`
- metrics: `outputs/Weather_train_metrics.csv`

## Inference

```bash
python run_infer.py
```

`run_infer.py` supports two modes:

- offline rolling evaluation when `infer.future_path` is set
- single-window forecasting when `infer.input_path` is used instead

Default offline outputs:

- predictions: `outputs/Weather_Prediction.csv`
- metrics: `outputs/Weather_Metrics.json`

## Benchmarks

Available baselines:

- `benchmarks/arima.py`
- `benchmarks/ets.py`
- `benchmarks/rnn.py`
- `benchmarks/tcn.py`

Run the benchmark suite:

```bash
python scripts/run_benchmark_suite.py --dry-run
```

Example full run:

```bash
python scripts/run_benchmark_suite.py ^
  --datasets Weather NASDAQ ^
  --benchmarks arima ets rnn tcn ^
  --device cuda
```

## Useful Scripts

Loss curve:

```bash
python scripts/loss_visualization.py ^
  --metrics-path outputs/Weather_train_metrics.csv ^
  --output-path outputs/Weather_loss_curve.png
```

Prediction metrics:

```bash
python scripts/prediction_metrics.py ^
  --prediction-path outputs/Weather_Prediction.csv ^
  --seasonality 1
```

Prediction plot:

```bash
python scripts/result_visualization.py ^
  --prediction-path outputs/Weather_Prediction.csv ^
  --window-size 200
```

## Notes

- `out_dim` and `tgt_dim` must match in the current Transformer pipeline.
- If you switch datasets, update `configs/default.yaml` accordingly.
- Existing files in `outputs/` and `checkpoints/` may be overwritten if you reuse the same paths.
