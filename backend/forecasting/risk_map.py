"""
Configurable risk-management signal-processing pipeline.

Deliberately kept SEPARATE from the existing Kelly-criterion-based
sizing in backend/forecasting/risk.py — that approach sizes positions
from a model's own real, validated win-rate/avg-win/avg-loss track
record. This is a different, explicitly experimental approach: a
continuous transform-and-clip pipeline operating directly on a
probability and a forecast-volatility estimate, with every stage
individually toggleable so its actual value can be measured, not
assumed.

Pipeline, in order:
  raw probability -> signal transform -> dead-zone -> volatility
  scaling -> exposure cap -> final position

Nothing here claims the dead-zone or volatility scaling IMPROVES
performance — that's exactly what the 5-way backtest comparison in
backend/backtesting/tcn_comparison.py exists to actually check.
"""
from __future__ import annotations

from dataclasses import dataclass

from backend.forecasting.tcn_model import probability_to_signal

ALL_STAGES = {"transform", "dead_zone", "vol_scaling", "exposure_cap"}


@dataclass
class RiskMapConfig:
    dead_zone_threshold: float = 0.1  # |signal| below this gets zeroed — small, likely-noise signals shouldn't auto-create a position
    target_volatility: float = 0.02  # the forecast-vol level at which scaling leaves the signal's magnitude unchanged
    min_volatility: float = 0.005  # floor — avoids dividing by a near-zero (or erroneous) volatility reading
    max_volatility: float = 0.10  # ceiling — avoids an extremely volatile stock crushing the position to near-zero
    max_position_pct: float = 0.20  # hard exposure cap, matches the existing MAX_POSITION_PCT convention (20%) elsewhere in the app
    vol_scaling_enabled: bool = True


def apply_dead_zone(signal: float, threshold: float) -> float:
    return 0.0 if abs(signal) < threshold else signal


def apply_volatility_scaling(signal: float, forecast_volatility: float | None, config: RiskMapConfig) -> float:
    """position ∝ signal / forecast_volatility, expressed as a scaling
    factor relative to a configured target volatility so the semantics
    stay intuitive: a stock at exactly target_volatility passes the
    signal through unchanged; calmer stocks get sized up (capped by
    the min_volatility floor); wilder stocks get sized down (capped by
    the max_volatility ceiling — never all the way to zero, since
    max_volatility is itself finite, not infinite)."""
    if forecast_volatility is None or not config.vol_scaling_enabled:
        return signal
    clamped_vol = max(config.min_volatility, min(config.max_volatility, forecast_volatility))
    return signal * (config.target_volatility / clamped_vol)


def apply_exposure_cap(position: float, config: RiskMapConfig) -> float:
    return max(-config.max_position_pct, min(config.max_position_pct, position))


def process_signal(probability_up: float, forecast_volatility: float | None,
                    config: RiskMapConfig | None = None, stages: set[str] | None = None) -> dict:
    """Runs the full pipeline, or a SUBSET of stages — this is what
    lets the 5-way backtest comparison reproduce each variant exactly:
    stages={"transform"} alone is "raw model signal", adding
    "dead_zone" is "+ neutral zone", etc. Returns every intermediate
    value, not just the final one, so the tradeoff at each stage is
    visible rather than hidden inside one opaque number."""
    config = config or RiskMapConfig()
    stages = ALL_STAGES if stages is None else stages

    raw_signal = float(probability_to_signal(probability_up)) if "transform" in stages else float(2 * probability_up - 1)
    after_dead_zone = apply_dead_zone(raw_signal, config.dead_zone_threshold) if "dead_zone" in stages else raw_signal
    after_vol_scaling = (apply_volatility_scaling(after_dead_zone, forecast_volatility, config)
                          if "vol_scaling" in stages else after_dead_zone)
    final_position = apply_exposure_cap(after_vol_scaling, config) if "exposure_cap" in stages else after_vol_scaling

    return {
        "raw_signal": round(raw_signal, 4),
        "after_dead_zone": round(after_dead_zone, 4),
        "after_vol_scaling": round(after_vol_scaling, 4),
        "final_position": round(final_position, 4),
    }
