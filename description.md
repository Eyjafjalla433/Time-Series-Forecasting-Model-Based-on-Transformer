# Project Architecture and Implementation

This document is written for a project defense. It explains what the project does, how the code is organized, how the Transformer model is built, and how the training and evaluation pipeline works.

## 1. Project Goal

This project is a PyTorch-based time-series forecasting system. Its core model is an Encoder-Decoder Transformer. Given a multivariate historical time window, the model predicts several future steps of the target variable.

The active default configuration uses the `Weather` dataset:

- Input length: `input_length = 12`
- Prediction length: `pred_length = 6`
- Input feature dimension: `src_dim = 21`
- Decoder target dimension: `tgt_dim = 1`
- Output dimension: `out_dim = 1`

The project also includes ARIMA, ETS, RNN, and TCN baselines, so the Transformer result can be compared with both traditional statistical methods and other neural forecasting models.

## 2. Overall Architecture

```text
Raw CSV data
   |
   v
data/preprocessing.py
   |  Convert TFB long-format data into a numeric [T, F] wide table
   |  and split it chronologically into train/future files
   v
data/processed/*_wide_train.csv, *_wide_future.csv
   |
   v
run_train.py
   |  Build sliding-window datasets, train the Transformer,
   |  and save the best checkpoint
   v
checkpoints/*.pt + outputs/*_train_metrics.csv
   |
   v
run_infer.py
   |  Load the checkpoint and run either single-window inference
   |  or rolling offline evaluation on future data
   v
outputs/*_Prediction.csv + outputs/*_Metrics.json
```

## 3. Directory Responsibilities

- `configs/`: experiment configuration. The default entry reads `configs/default.yaml`.
- `data/`: data preprocessing and sliding-window dataset construction.
- `models/`: Transformer model code, including embeddings, attention, encoder layers, and decoder layers.
- `engine/`: training loop, inference helpers, and evaluation utilities.
- `utils/`: YAML config loading and checkpoint save/load helpers.
- `benchmarks/`: ARIMA, ETS, RNN, and TCN baseline implementations.
- `scripts/`: metrics, visualization, and benchmark-suite helper scripts.
- `checkpoints/`: saved model weights.
- `outputs/`: predictions, metric files, training logs, and generated figures.

## 4. Data Flow and Window Construction

The main pipeline uses a numeric wide CSV table with shape `[T, F]`:

- `T` is the number of time steps.
- `F` is the number of features at each time step.
- The first `src_dim` columns are used as encoder inputs.
- The last `tgt_dim` columns are used as decoder target inputs.
- The last `out_dim` columns are used as prediction outputs.

`TimeSeriesWindowDataset` converts the continuous table into supervised samples:

```text
src      = historical input_length steps
tgt_full = the last source step plus the future pred_length steps
y        = the true future pred_length steps
```

`tgt_full` has one more step than `y` because the decoder is trained with shifted targets:

- Decoder input: `tgt_full[:, :-1, :]`
- Supervision label: `tgt_full[:, 1:, :]`

This means the model learns to predict the next value at each decoder position. That training setup matches the autoregressive inference stage, where the model must generate future values step by step.

## 5. How the Transformer Model Is Built

The model is defined in `models/model.py`. The high-level structure is:

```text
src -> TimeSeriesEmbedding -> Encoder -> memory
tgt -> TimeSeriesEmbedding -> Decoder(memory) -> hidden states
hidden states -> Linear generator -> predictions
```

The model is created by `make_model(...)`, which instantiates `TransformerTimeSeriesModel` and initializes weight matrices with Xavier uniform initialization.

### 5.1 Input Projection

Raw time-series features are continuous values, not discrete word tokens. Therefore, the project does not use an NLP-style token embedding table. Instead, `TimeSeriesEmbedding` uses a linear layer:

```text
[B, L, input_dim] -> [B, L, d_model]
```

Here:

- `B` is batch size.
- `L` is sequence length.
- `input_dim` is either `src_dim` or `tgt_dim`.
- `d_model` is the Transformer hidden size.

After the linear projection, sinusoidal positional encoding is added. This is important because attention itself does not know the order of time steps; positional encoding gives the model information about sequence order.

### 5.2 Encoder

The encoder receives the embedded historical window:

```text
src: [B, input_length, src_dim]
src_embed(src): [B, input_length, d_model]
```

It passes this tensor through `N` identical `EncoderLayer` blocks. Each encoder layer contains:

1. Multi-head self-attention over the historical window.
2. A position-wise feed-forward network.
3. Residual connections and layer normalization around both sublayers.

The encoder output is called `memory`. It is the model's learned representation of the historical context.

### 5.3 Multi-Head Attention

`MultiHeadedAttention` first projects the input into query, key, and value tensors. It then splits the hidden dimension into `h` heads:

```text
[B, L, d_model] -> [B, h, L, d_k]
```

Each head computes scaled dot-product attention:

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
```

Using multiple heads lets the model focus on different temporal relationships at the same time, such as short-term changes, repeated patterns, or longer dependencies.

### 5.4 Decoder

The decoder receives shifted target inputs during training:

```text
tgt: [B, pred_length, tgt_dim]
tgt_embed(tgt): [B, pred_length, d_model]
```

Each decoder layer has three sublayers:

1. Masked self-attention over the decoder sequence.
2. Cross-attention from decoder states to encoder `memory`.
3. A position-wise feed-forward network.

The masked self-attention uses a causal mask. This prevents the decoder position at step `t` from seeing future decoder steps. That design is essential because real forecasting cannot use future ground-truth values.

The cross-attention layer connects the predicted future sequence to the encoded historical window. In simple terms, the decoder asks: "given what the encoder learned from history, what should the next future values look like?"

### 5.5 Output Layer

The decoder returns hidden states with shape:

```text
[B, pred_length, d_model]
```

The final `generator` is a linear layer:

```text
d_model -> out_dim
```

So the final output shape is:

```text
[B, pred_length, out_dim]
```

For the current configuration, `out_dim = 1`, so each future time step predicts one target value.

## 6. Training Flow

The training entry point is `run_train.py`:

1. Load `configs/default.yaml`.
2. Load the training CSV. If no CSV is configured, generate a synthetic series for pipeline testing.
3. Split the series chronologically into training and validation parts.
4. Build sliding-window samples with `TimeSeriesWindowDataset`.
5. Create the Encoder-Decoder Transformer with `make_model`.
6. Use Adam optimization with warmup plus cosine learning-rate decay.
7. Train for multiple epochs and evaluate on the validation set after each epoch.
8. Save the checkpoint with the lowest validation loss.
9. Write per-epoch training and validation losses to a CSV file.

For the defense, one important point is that time-series data should not be randomly split. This project uses chronological splitting to reduce future information leakage. The validation series keeps only the necessary input-window overlap so the first validation window has enough historical context.

## 7. Inference and Offline Evaluation

The inference entry point is `run_infer.py`. It supports two modes:

- Rolling offline evaluation when `infer.future_path` is configured.
- Single-window forecasting when `infer.future_path` is not configured and `infer.input_path` is used.

Rolling offline evaluation works as follows:

1. Use the train CSV as historical context.
2. Use the future CSV as held-out future ground truth.
3. Select one prediction origin every `eval_stride` steps.
4. Take the `input_length` rows before the origin as the source window.
5. Autoregressively predict `pred_length` future steps.
6. Compare predictions with the true future values and compute MSE, RMSE, MAE, and MAPE.

The autoregressive function is `engine/infer.py::autoregressive_forecast`:

```text
create the first decoder token
predict step 1
append the step-1 prediction to the decoder input
predict step 2
repeat until pred_length steps are generated
```

This matches the real forecasting setting because future ground-truth values are not available at inference time.

## 8. Baseline Experiments

The `benchmarks/` directory contains four comparison methods:

- `arima.py`: ARIMA statistical model for a single target column.
- `ets.py`: exponential smoothing model for level, trend, and seasonality.
- `rnn.py`: recurrent neural network baseline.
- `tcn.py`: Temporal Convolutional Network baseline.

`scripts/run_benchmark_suite.py` can run multiple datasets and multiple baselines under a unified output format.

In the defense, this supports the argument that the Transformer is evaluated in a broader experimental framework, not as an isolated model.

## 9. Main Outputs

- `checkpoints/Weather_model.pt`: best Transformer checkpoint.
- `outputs/Weather_train_metrics.csv`: training and validation loss per epoch.
- `outputs/Weather_Prediction.csv`: rolling prediction results.
- `outputs/Weather_Metrics.json`: offline evaluation metrics.
- `outputs/benchmark_suite/`: benchmark-suite outputs.

## 10. Suggested Defense Explanation Order

1. Define the problem: predict future target values from multivariate historical sequences.
2. Explain preprocessing: convert long-format data to a wide table, move target columns to the end, and split data chronologically.
3. Explain the model: the encoder reads history, the decoder generates the future, and multi-head attention captures different temporal dependencies.
4. Explain training: sliding windows, shifted decoder targets, MSE loss, validation-based checkpoint selection.
5. Explain evaluation: rolling offline evaluation, forecasting metrics, and comparison with ARIMA, ETS, RNN, and TCN.

One-sentence summary:

> This project implements a complete time-series forecasting framework, covering preprocessing, Transformer modeling, training, rolling evaluation, baseline comparison, and visualization.
