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
    base = _norm(row.get("Potencial_total"), 15.0)
    p85 = _norm(row.get("Potencial_o85"), 100.0)
    p95 = _norm(row.get("corners_o95_potential"), 100.0)
    p105 = _norm(row.get("corners_o105_potential"), 100.0)
    odds = row.get("ODDS") or row.get("Odds_O85") or row.get("odds_corners_over_85")
    odds = float(odds) if odds and odds > 1 else np.nan

    score = (
        -2.0
        + 3.2 * p85
        + 1.5 * base
        + 1.1 * p95
        + 0.5 * p105
    )
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
