"""
Model C — LSTM (spec section 9), for capturing sequential/temporal
patterns that tabular models (Model A/B) treat each row independently
and can't see (e.g. "volatility has been climbing for 8 straight days"
is a shape in time, not a single-row feature).

PyTorch is a genuinely heavy, sometimes finicky dependency — this
module is fully optional. If torch isn't installed, TORCH_AVAILABLE is
False and every function here either no-ops or raises a clear
ImportError that the service layer catches and reports honestly to the
UI ("LSTM unavailable — PyTorch not installed") rather than crashing
the whole prediction pipeline.

The sequence-windowing logic (build_sequences) is pure NumPy and has
zero torch dependency, so it works — and is tested — even in
environments without PyTorch installed.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    # Explicitly cap torch's own intra-op thread pool at 1, as a
    # programmatic backup to the OMP_NUM_THREADS env var set in
    # app.py. Found via real testing: PyTorch's thread pool can
    # silently deadlock on macOS when it initializes in a process
    # where other libraries (numpy/scikit-learn/XGBoost) already set
    # up their own BLAS/OpenMP threading — this happened even with the
    # env var set in some configurations, so belt-and-suspenders here.
    torch.set_num_threads(1)
    TORCH_AVAILABLE = True
except Exception:  # noqa: BLE001
    # Deliberately broad, not just ImportError — same reasoning as
    # xgboost's import guard in models.py: a broken/partial native
    # install can fail with something other than a clean ImportError,
    # and that must not be allowed to crash the whole app at startup.
    TORCH_AVAILABLE = False


def build_sequences(X: np.ndarray, y: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Converts flat (n_rows, n_features) data into rolling windows of
    shape (n_samples, seq_len, n_features), where y_seq[i] is the
    target immediately following the i-th window — i.e. window i covers
    rows [i, i+seq_len) and predicts the target at row i+seq_len-1
    (which is y[i+seq_len-1], already the forward-looking target
    computed upstream in features.py). No torch dependency — pure
    NumPy, so this is testable without PyTorch installed."""
    n = len(X)
    if n <= seq_len:
        return np.empty((0, seq_len, X.shape[1])), np.empty((0,))

    n_samples = n - seq_len + 1
    X_seq = np.empty((n_samples, seq_len, X.shape[1]), dtype=np.float64)
    y_seq = np.empty((n_samples,), dtype=np.float64)
    for i in range(n_samples):
        X_seq[i] = X[i:i + seq_len]
        y_seq[i] = y[i + seq_len - 1]
    return X_seq, y_seq


if TORCH_AVAILABLE:
    class _LSTMNet(nn.Module):
        """LSTM with a self-attention layer over its own past outputs —
        adapted from a reviewed tutorial's "LSTM with Attention" pattern
        (query = current timestep, key/value = every past timestep's
        LSTM output, combined via residual connection).

        Why this matters: nn.LSTM's `out[:, -1, :]` — what the previous
        version used exclusively — only reflects earlier days *insofar
        as the recurrent connections successfully carried that
        information forward* through the hidden-state bottleneck.
        Attention gives the model a second, direct path: the most
        recent day can explicitly query every one of the past `seq_len`
        days' LSTM outputs and pull specific information from whichever
        ones are actually relevant to this specific prediction, rather
        than being limited to whatever survived being compressed
        through the recurrence alone.

        num_heads must evenly divide hidden_size — nn.MultiheadAttention
        requires this."""
        def __init__(self, n_features: int, hidden_size: int = 32, num_layers: int = 1, num_heads: int = 4):
            super().__init__()
            self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size,
                                 num_layers=num_layers, batch_first=True)
            self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
            self.head = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.lstm(x)  # (batch, seq_len, hidden_size) — every timestep's output, not just the last
            query = out[:, -1:, :]  # (batch, 1, hidden_size) — only today queries; keys/values span the whole window
            attn_out, _ = self.attention(query, out, out)
            combined = attn_out.squeeze(1) + query.squeeze(1)  # residual: attention augments, doesn't replace, the LSTM's own read
            return self.head(combined).squeeze(-1)


class LSTMModel:
    """Thin wrapper matching the same fit()/predict() interface as the
    sklearn/XGBoost models, so it can slot into the same service-layer
    code — internally it windows flat data into sequences, trains a
    small LSTM on CPU (kept deliberately small: 1 layer, 32 hidden
    units, ~20 epochs — this is a personal local research tool running
    on a laptop CPU, not a GPU cluster), and predicts from the most
    recent window."""

    def __init__(self, seq_len: int = 20, hidden_size: int = 32, epochs: int = 20, lr: float = 0.01):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not installed — LSTM model unavailable")
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.lr = lr
        self.model: Optional["_LSTMNet"] = None
        self.feature_mean = None
        self.feature_std = None

    def _normalize(self, X: np.ndarray, fit: bool) -> np.ndarray:
        if fit:
            self.feature_mean = X.mean(axis=0)
            self.feature_std = X.std(axis=0)
            self.feature_std[self.feature_std == 0] = 1.0
        return (X - self.feature_mean) / self.feature_std

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LSTMModel":
        X_norm = self._normalize(X, fit=True)
        X_seq, y_seq = build_sequences(X_norm, y, self.seq_len)
        if len(X_seq) < 10:
            raise ValueError(f"Not enough rows ({len(X)}) to build LSTM training sequences of length {self.seq_len}")

        self.model = _LSTMNet(n_features=X.shape[1], hidden_size=self.hidden_size)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()

        X_tensor = torch.tensor(X_seq, dtype=torch.float32)
        y_tensor = torch.tensor(y_seq, dtype=torch.float32)

        self.model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            preds = self.model(X_tensor)
            loss = loss_fn(preds, y_tensor)
            loss.backward()
            optimizer.step()

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts one value per input row, using that row and the
        `seq_len - 1` rows before it as context. Rows without enough
        preceding history (the first `seq_len - 1` rows of whatever
        array is passed in) get a prediction of 0.0 (neutral) rather
        than an error or a fabricated extrapolation — the walk-forward
        harness's test blocks are short, so this matters for the first
        few predictions of each block, which is why the harness passes
        enough trailing context in practice (see PredictionService)."""
        if self.model is None:
            raise RuntimeError("LSTMModel.predict() called before fit()")
        X_norm = self._normalize(X, fit=False)

        self.model.eval()
        preds = np.zeros(len(X))
        with torch.no_grad():
            for i in range(len(X)):
                start = max(0, i - self.seq_len + 1)
                window = X_norm[start:i + 1]
                if len(window) < self.seq_len:
                    continue  # not enough history yet — leave as neutral (0.0)
                x_tensor = torch.tensor(window[np.newaxis, :, :], dtype=torch.float32)
                preds[i] = self.model(x_tensor).item()
        return preds


def make_lstm(seq_len: int = 20, epochs: int = 20):
    return LSTMModel(seq_len=seq_len, epochs=epochs)
