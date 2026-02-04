from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import pandas as pd


def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def _norm(value, scale: float) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 0.0
    try:
        return float(value) / scale
    except (ValueError, TypeError):
        return 0.0


def _score_row(row: pd.Series) -> Tuple[float, float]:
    total_pot = _norm(row.get("Potencial_total"), 15.0)
    over95 = _norm(row.get("corners_o95_potential"), 100.0)
    over105 = _norm(row.get("corners_o105_potential"), 100.0)
    under_frac = 1.0 - over105
    odds = row.get("Odds_U105") or row.get("odds_corners_under_105")
    odds = float(odds) if odds and odds > 1 else np.nan

    score = 1.8 + 2.4 * under_frac - 1.3 * total_pot - 0.9 * over95
    prob = _sigmoid(score)
    ev = prob * odds - 1 if odds and not math.isnan(odds) else np.nan
    return prob, ev


def aplicar_modelo(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        df["Prob_modelo"] = pd.Series(dtype=float)
        df["EV_modelo"] = pd.Series(dtype=float)
        return df

    probs, evs = zip(*[_score_row(row) for _, row in df.iterrows()])
    df = df.copy()
    df["Prob_modelo"] = np.round(probs, 3)
    df["EV_modelo"] = np.round(evs, 3)
    return df
