"""
Model D — causal Temporal Convolutional Network (TCN).

Like backend/forecasting/lstm_model.py, this is fully optional —
PyTorch is a heavy dependency, so if it isn't installed, TORCH_AVAILABLE
is False and the service layer reports "TCN unavailable" honestly
rather than crashing the pipeline.

The one property that matters more than anything else here: causality.
A "causal" convolution must never let information from a future
timestep reach the prediction for an earlier one. This is easy to get
wrong — PyTorch's default Conv1d padding is SYMMETRIC (padding added to
both sides), which silently leaks future timesteps into the receptive
field of earlier ones. This module manually left-pads and trims the
right side after each convolution instead, so every output at position
t depends only on inputs at positions <= t. This is proven, not just
asserted — see tests/test_tcn_model.py, which perturbs only a future
timestep and confirms an earlier position's output doesn't change by
even a floating-point epsilon.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    torch.set_num_threads(1)  # see lstm_model.py for why this matters on macOS
    TORCH_AVAILABLE = True
except Exception:  # noqa: BLE001
    TORCH_AVAILABLE = False


def build_windows(X: np.ndarray, y: np.ndarray, window_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Converts flat (n_rows, n_features) data into rolling windows of
    shape (n_samples, window_len, n_features) — window i covers rows
    [i, i+window_len) and predicts the target already computed for row
    i+window_len-1 (the forward-looking label built upstream in
    features.py; this function itself looks only backward). Pure
    NumPy, no torch dependency, so it's testable without PyTorch."""
    n = len(X)
    if n <= window_len:
        return np.empty((0, window_len, X.shape[1])), np.empty((0,))

    n_samples = n - window_len + 1
    X_win = np.empty((n_samples, window_len, X.shape[1]), dtype=np.float64)
    y_win = np.empty((n_samples,), dtype=np.float64)
    for i in range(n_samples):
        X_win[i] = X[i:i + window_len]
        y_win[i] = y[i + window_len - 1]
    return X_win, y_win


if TORCH_AVAILABLE:
    class _CausalConv1d(nn.Module):
        """A Conv1d that only ever looks backward in time. Left-pads
        the input by (kernel_size - 1) * dilation, convolves with NO
        built-in padding, so the output naturally has the same length
        as the input with zero right-side leakage — there's no
        "trim the right edge" step needed because nothing ever gets
        added there in the first place."""
        def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
            super().__init__()
            self.left_pad = (kernel_size - 1) * dilation
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                                   padding=0, dilation=dilation)

        def forward(self, x):
            # x: (batch, channels, time). Pad only the LEFT (past) side.
            x = F.pad(x, (self.left_pad, 0))
            return self.conv(x)

    class _TCNBlock(nn.Module):
        """One residual block: two causal convolutions with the same
        dilation, ELU activations, dropout, and a residual connection
        (with a 1x1 conv to match channel counts if they differ)."""
        def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
            super().__init__()
            self.conv1 = _CausalConv1d(in_channels, out_channels, kernel_size, dilation)
            self.conv2 = _CausalConv1d(out_channels, out_channels, kernel_size, dilation)
            self.dropout = nn.Dropout(dropout)
            self.act = nn.ELU()
            self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

        def forward(self, x):
            out = self.act(self.conv1(x))
            out = self.dropout(out)
            out = self.act(self.conv2(out))
            out = self.dropout(out)
            residual = x if self.downsample is None else self.downsample(x)
            return self.act(out + residual)

    class _TCNNet(nn.Module):
        """Stacks num_layers residual blocks with an exponentially
        growing dilation schedule (1, 2, 4, 8, ...) — standard TCN
        design, giving later layers an exponentially larger receptive
        field without needing more layers or a bigger kernel. Ends
        with a 1x1 conv down to a single channel, takes only the FINAL
        (most recent) timestep, and applies a sigmoid to produce
        P(next bar return > 0)."""
        def __init__(self, n_features: int, num_layers: int = 4, kernel_size: int = 3,
                     hidden_channels: int = 16, dropout: float = 0.2):
            super().__init__()
            layers = []
            in_ch = n_features
            for i in range(num_layers):
                dilation = 2 ** i
                layers.append(_TCNBlock(in_ch, hidden_channels, kernel_size, dilation, dropout))
                in_ch = hidden_channels
            self.blocks = nn.Sequential(*layers)
            self.head = nn.Conv1d(hidden_channels, 1, 1)

        def forward(self, x):
            # x arrives as (batch, seq_len, n_features) — Conv1d wants
            # channels second: (batch, n_features, seq_len).
            x = x.transpose(1, 2)
            out = self.blocks(x)
            out = self.head(out)  # (batch, 1, seq_len)
            last = out[:, :, -1]  # only the most recent timestep's output — same "today queries" principle as the LSTM
            return torch.sigmoid(last).squeeze(-1)


class TCNModel:
    """Same fit()/predict() interface as every other model in this
    app, so it slots into the identical service-layer code without
    special-casing. Predicts P(next bar return > 0) rather than a
    direct return value — the ensemble/service layer maps this to a
    signed prediction via probability_to_signal() below, matching
    the sign convention every other model already uses."""

    def __init__(self, window_len: int = 64, num_layers: int = 4, kernel_size: int = 3,
                 hidden_channels: int = 16, dropout: float = 0.2, epochs: int = 20, lr: float = 0.001):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not installed — TCN model unavailable")
        self.window_len = window_len
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.hidden_channels = hidden_channels
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.model: Optional["_TCNNet"] = None
        self.feature_mean = None
        self.feature_std = None

    def _normalize(self, X: np.ndarray, fit: bool) -> np.ndarray:
        if fit:
            self.feature_mean = X.mean(axis=0)
            self.feature_std = X.std(axis=0)
            self.feature_std[self.feature_std == 0] = 1.0
        return (X - self.feature_mean) / self.feature_std

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TCNModel":
        X_norm = self._normalize(X, fit=True)
        X_win, y_return = build_windows(X_norm, y, self.window_len)
        if len(X_win) < 10:
            raise ValueError(f"Not enough rows ({len(X)}) to build TCN windows of length {self.window_len}")

        # The TCN predicts P(up) — convert the continuous return target
        # to a binary "did it go up" label ONLY for training the
        # probability head; the raw return itself is never used as an
        # input to future rows (that would be leakage), just as a label.
        y_binary = (y_return > 0).astype(np.float32)

        self.model = _TCNNet(n_features=X.shape[1], num_layers=self.num_layers, kernel_size=self.kernel_size,
                              hidden_channels=self.hidden_channels, dropout=self.dropout)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = torch.nn.BCELoss()

        X_tensor = torch.tensor(X_win, dtype=torch.float32)
        y_tensor = torch.tensor(y_binary, dtype=torch.float32)

        self.model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            probs = self.model(X_tensor)
            loss = loss_fn(probs, y_tensor)
            loss.backward()
            optimizer.step()

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns P(next bar return > 0) per row, using that row and
        the window_len - 1 rows before it as context. Rows without
        enough preceding history get 0.5 (genuinely uninformative —
        the honest "I don't know" for a probability, unlike 0.0 which
        would falsely imply high-confidence bearishness)."""
        if self.model is None:
            raise RuntimeError("TCNModel.predict_proba() called before fit()")
        X_norm = self._normalize(X, fit=False)

        self.model.eval()
        probs = np.full(len(X), 0.5)
        with torch.no_grad():
            for i in range(len(X)):
                start = max(0, i - self.window_len + 1)
                window = X_norm[start:i + 1]
                if len(window) < self.window_len:
                    continue
                x_tensor = torch.tensor(window[np.newaxis, :, :], dtype=torch.float32)
                probs[i] = self.model(x_tensor).item()
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Matches every other model's predict() interface (returns a
        continuous value the ensemble can average with returns from
        other models) by converting probability to a signed signal —
        see probability_to_signal(). Rows with no prediction (not
        enough history) correctly map to signal 0.0 (neutral), since
        probability_to_signal(0.5) == 0.0."""
        probs = self.predict_proba(X)
        return probability_to_signal(probs)


def probability_to_signal(probability_up) -> np.ndarray:
    """signal = 2 * P(up) - 1, mapping [0, 1] to [-1, +1]. Configurable
    in spirit (a different transform could be swapped in), but this is
    the standard, simplest mapping and matches exactly what was
    requested. P(up)=0.5 (genuinely uninformative) correctly maps to
    signal=0.0 (neutral) — the two "I don't know" states line up."""
    return 2 * np.asarray(probability_up) - 1


def make_tcn(window_len: int = 64, num_layers: int = 4, kernel_size: int = 3,
             hidden_channels: int = 16, dropout: float = 0.2, epochs: int = 20):
    return TCNModel(window_len=window_len, num_layers=num_layers, kernel_size=kernel_size,
                     hidden_channels=hidden_channels, dropout=dropout, epochs=epochs)
