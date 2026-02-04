import math
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from app.ingest.normalizacion.two_way import fair_prob_two_way as two_way


def _poisson_over15_from_sum(lam_sum: float) -> Optional[float]:
    """P(X >= 2) para X ~ Poisson(lam_sum)."""
    if lam_sum is None or lam_sum <= 0:
        return None
    return 1.0 - math.exp(-lam_sum) * (1.0 + lam_sum)


def _to_float(x) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
        if v <= 0:
            return None
        return v
    except (TypeError, ValueError):
        return None


def _first_float(d: Dict[str, Any], keys: List[str]) -> Optional[float]:
    for k in keys:
        if k in d:
            val = _to_float(d.get(k))
            if val is not None:
                return val
    return None


def _pex_ev_norm(p: Optional[float], odd: Optional[float]) -> float:
    if p is None or odd is None:
        return np.nan
    if np.isnan(p) or np.isnan(odd) or odd <= 1.0:
        return np.nan
    ev = p * odd - 1.0
    denom = odd - 1.0
    if denom <= 0:
        return np.nan
    return float(np.clip(ev / denom, 0.0, 1.0))


def over15_to_df(normalized: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for n in normalized:
        lam_a = float(n.get("team_a_xg_prematch") or n.get("home_xg_prematch") or 0.0)
        lam_b = float(n.get("team_b_xg_prematch") or n.get("away_xg_prematch") or 0.0)
        lam_sum = lam_a + lam_b if (lam_a > 0 and lam_b > 0) else None

        odds_over = _first_float(n, ["odds_ft_over15", "odds_over15", "odds_o15"])
        odds_under = _first_float(n, ["odds_ft_under15", "odds_under15", "odds_u15"])

        p_model_over = _poisson_over15_from_sum(lam_sum) if lam_sum else None
        p_fair_over = two_way(odds_over, odds_under) if (odds_over and odds_under) else None

        edge_over = (p_model_over - p_fair_over) if (p_model_over is not None and p_fair_over is not None) else None
        ev_over = (p_model_over * float(odds_over) - 1) if (p_model_over is not None and odds_over) else None

        o15_potential = n.get("o15_potential")
        if o15_potential is None and p_model_over is not None:
            o15_potential = round(p_model_over * 100, 2)

        rows.append({
            "match_id": n.get("match_id"),
            "season_id": n.get("season_id"),
            "season_label": n.get("season_label"),
            "home_id": n.get("home_id"),
            "away_id": n.get("away_id"),
            "home": n.get("home"),
            "away": n.get("away"),
            "competition_id": n.get("competition_id"),
            "kickoff_cdmx": n.get("kickoff_local_cdmx"),
            "odds_over15": odds_over,
            "odds_under15": odds_under,
            "p_model_over15": round(p_model_over, 4) if p_model_over is not None else None,
            "p_fair_over15": round(p_fair_over, 4) if p_fair_over is not None else None,
            "edge_over15": round(edge_over, 4) if edge_over is not None else None,
            "ev_over15": round(ev_over, 4) if ev_over is not None else None,
            "o15_potential": o15_potential,
            "cards_potential": n.get("cards_potential"),
        })

    df = pd.DataFrame(rows)

    df["cuota_real_over15"] = pd.to_numeric(df.get("odds_over15"), errors="coerce")
    df["cuota_model_over15"] = df["p_model_over15"].apply(lambda x: (1/x) if pd.notna(x) and x > 0 else np.nan)
    df["cuota_justa_over15"] = df["p_fair_over15"].apply(lambda x: (1/x) if pd.notna(x) and x > 0 else np.nan)

    df["value_pct_model_over15"] = np.where(
        df["cuota_real_over15"].notna() & df["cuota_model_over15"].notna(),
        (df["cuota_real_over15"] - df["cuota_model_over15"]) / df["cuota_model_over15"] * 100,
        np.nan
    )
    df["value_pct_justa_over15"] = np.where(
        df["cuota_real_over15"].notna() & df["cuota_justa_over15"].notna(),
        (df["cuota_real_over15"] - df["cuota_justa_over15"]) / df["cuota_justa_over15"] * 100,
        np.nan
    )

    df["p_model_under15"] = np.where(df["p_model_over15"].notna(), 1 - df["p_model_over15"], np.nan)
    df["p_fair_under15"] = np.where(df["p_fair_over15"].notna(), 1 - df["p_fair_over15"], np.nan)

    df["cuota_real_under15"] = pd.to_numeric(df.get("odds_under15"), errors="coerce")
    df["cuota_model_under15"] = df["p_model_under15"].apply(lambda x: (1/x) if pd.notna(x) and x > 0 else np.nan)
    df["cuota_justa_under15"] = df["p_fair_under15"].apply(lambda x: (1/x) if pd.notna(x) and x > 0 else np.nan)

    df["edge_under15"] = np.where(
        df["p_model_under15"].notna() & df["p_fair_under15"].notna(),
        df["p_model_under15"] - df["p_fair_under15"],
        np.nan
    )
    df["ev_under15"] = np.where(
        df["p_model_under15"].notna() & df["cuota_real_under15"].notna(),
        df["p_model_under15"] * df["cuota_real_under15"] - 1,
        np.nan
    )

    df["value_pct_model_under15"] = np.where(
        df["cuota_real_under15"].notna() & df["cuota_model_under15"].notna(),
        (df["cuota_real_under15"] - df["cuota_model_under15"]) / df["cuota_model_under15"] * 100,
        np.nan
    )
    df["value_pct_justa_under15"] = np.where(
        df["cuota_real_under15"].notna() & df["cuota_justa_under15"].notna(),
        (df["cuota_real_under15"] - df["cuota_justa_under15"]) / df["cuota_justa_under15"] * 100,
        np.nan
    )

    df["pex_hit_over15"] = df["p_model_over15"]
    df["pex_ev_norm_over15"] = [
        _pex_ev_norm(p, o) for p, o in zip(df["p_model_over15"], df["cuota_real_over15"])
    ]

    df["pex_hit_under15"] = df["p_model_under15"]
    df["pex_ev_norm_under15"] = [
        _pex_ev_norm(p, o) for p, o in zip(df["p_model_under15"], df["cuota_real_under15"])
    ]

    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].round(3)

    return df
