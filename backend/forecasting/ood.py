"""
Out-of-distribution (OOD) detection — this exists directly because of
a real lesson from testing: tree-based models can't extrapolate beyond
the value ranges they were trained on (see the Random Forest raw-price
extrapolation failure discussed during development — a model trained
on a stock trading $170-190 will structurally fail to predict $240).

Predicting RETURNS instead of raw price (which this app does) avoids
the worst of that, but the underlying risk doesn't fully disappear: if
today's market conditions are genuinely unlike anything in the
training history — a volatility spike far beyond anything seen before,
a relative-strength reading with no precedent — every model in the
ensemble is, to some degree, guessing outside its own experience. This
module doesn't stop a prediction from being made; it just says so
plainly when conditions warrant it, the same way a good analyst would
flag "we're in uncharted territory here."
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# A z-score beyond this magnitude is treated as "unusual enough to flag."
# 3 standard deviations is a standard, conservative threshold — under
# a normal distribution, only ~0.3% of values would fall this far out,
# so flagging at this level catches genuine outliers without being
# triggered by everyday noise.
ZSCORE_FLAG_THRESHOLD = 3.0


def compute_training_distribution(X: pd.DataFrame) -> dict:
    """Captures each feature's mean/std from the training set — this is
    the reference distribution the live prediction gets checked against."""
    return {
        "mean": X.mean().to_dict(),
        "std": X.std().replace(0, np.nan).to_dict(),  # avoid div-by-zero for a constant column
    }


def check_out_of_distribution(latest_row: pd.Series, training_distribution: dict) -> dict:
    """Compares the current (live) feature row against the training
    distribution and flags any feature that's an extreme outlier
    relative to what the model actually learned from."""
    means = training_distribution["mean"]
    stds = training_distribution["std"]

    flagged = []
    for feature, value in latest_row.items():
        mean = means.get(feature)
        std = stds.get(feature)
        if mean is None or std is None or std != std or pd.isna(value):  # std != std catches NaN
            continue
        z = (value - mean) / std
        if abs(z) >= ZSCORE_FLAG_THRESHOLD:
            flagged.append({
                "feature": feature,
                "current_value": round(float(value), 4),
                "training_mean": round(float(mean), 4),
                "z_score": round(float(z), 2),
            })

    flagged.sort(key=lambda f: abs(f["z_score"]), reverse=True)

    if not flagged:
        return {"is_out_of_distribution": False, "flagged_features": [], "warning": None}

    top = flagged[0]
    warning = (
        f"Current conditions look unusual compared to this model's training history — "
        f"'{top['feature']}' is {abs(top['z_score']):.1f} standard deviations from its typical "
        f"range (current: {top['current_value']}, typical: ~{top['training_mean']}). "
        f"The model is extrapolating somewhat beyond what it has actually seen before; "
        f"treat this prediction with extra caution."
    )

    return {"is_out_of_distribution": True, "flagged_features": flagged, "warning": warning}
