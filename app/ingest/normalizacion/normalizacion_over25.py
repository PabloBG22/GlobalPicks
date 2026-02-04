# app/ingest/normalizacion/normalizacion_over25.py
import math
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from app.ingest.normalizacion.two_way import fair_prob_two_way as two_way

def _poisson_over25_from_sum(lam_sum: float) -> Optional[float]:
    """P(X >= 3) para X ~ Poisson(lam_sum)."""
    if lam_sum is None or lam_sum <= 0:
        return None
    return 1.0 - math.exp(-lam_sum) * (1.0 + lam_sum + (lam_sum**2)/2.0)

def _to_float(x) -> Optional[float]:
    """Convierte a float, devolviendo None si es None/''/0/'0' o no convertible."""
    if x is None:
        return None
    try:
        v = float(x)
        if v <= 0:
            return None
        return v
    except (TypeError, ValueError):
        return None

def _first_float(d: Dict[str, Any], keys: list[str]) -> Optional[float]:
    """Devuelve el primer valor convertible>0 encontrado en d para cualquier key."""
    for k in keys:
        if k in d:
            val = _to_float(d.get(k))
            if val is not None:
                return val
    return None

def _pex_ev_norm(p: Optional[float], odd: Optional[float]) -> float:
    """
    EV normalizado a [0,1]:
      EV = p*odd - 1
      EV_max = odd - 1
      pex_ev_norm = clip(EV / (odd - 1), 0, 1)
    Devuelve NaN si faltan datos o odd <= 1.
    """
    if p is None or odd is None:
        return np.nan
    if np.isnan(p) or np.isnan(odd) or odd <= 1.0:
        return np.nan
    ev = p * odd - 1.0
    denom = odd - 1.0
    if denom <= 0:
        return np.nan
    return float(np.clip(ev / denom, 0.0, 1.0))

def over25_to_df(normalized: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for n in normalized:
        lam_a = float(n.get("team_a_xg_prematch") or n.get("home_xg_prematch") or 0.0)
        lam_b = float(n.get("team_b_xg_prematch") or n.get("away_xg_prematch") or 0.0)
        lam_sum = lam_a + lam_b if (lam_a > 0 and lam_b > 0) else None

        # Cuotas mercado Over/Under 2.5 (alias robustos)
        odds_over = _first_float(n, ["odds_ft_over25", "odds_over25", "odds_o25"])
        odds_under = _first_float(n, ["odds_ft_under25", "odds_under25", "odds_u25"])

        # Modelo
        p_model_over = _poisson_over25_from_sum(lam_sum) if lam_sum else None

        # Prob. justa (si hay ambas cuotas)
        p_fair_over = two_way(odds_over, odds_under) if (odds_over and odds_under) else None

        # Edge / EV (Over)
        edge_over = (p_model_over - p_fair_over) if (p_model_over is not None and p_fair_over is not None) else None
        ev_over = (p_model_over * float(odds_over) - 1) if (p_model_over is not None and odds_over) else None

        # Potencial (si no lo traes, usa el modelo)
        o25_potential = n.get("o25_potential")
        if o25_potential is None:
            o25_potential = round(p_model_over * 100, 2) if p_model_over is not None else None

        rows.append({
            "match_id" : n.get("match_id"),
            "season_id": n.get("season_id"),
            "season_label": n.get("season_label"),
            "home_id": n.get("home_id"),
            "away_id": n.get("away_id"),
            "home": n.get("home"),
            "away": n.get("away"),
            "competition_id": n.get("competition_id"),
            "kickoff_cdmx": n.get("kickoff_local_cdmx"),

            # cuotas crudas
            "odds_over25": odds_over,
            "odds_under25": odds_under,

            # métricas Over 2.5
            "p_model_over25": round(p_model_over, 4) if p_model_over is not None else None,
            "p_fair_over25": round(p_fair_over, 4) if p_fair_over is not None else None,
            "edge_over25": round(edge_over, 4) if edge_over is not None else None,
            "ev_over25": round(ev_over, 4) if ev_over is not None else None,

            "o25_potential": o25_potential,
            "cards_potential": n.get("cards_potential"),
        })

    df = pd.DataFrame(rows)

    # ===== Cuotas reales y de referencia (OVER) =====
    df["cuota_real_over25"]  = pd.to_numeric(df.get("odds_over25"), errors="coerce")
    df["cuota_model_over25"] = df["p_model_over25"].apply(lambda x: (1/x) if pd.notna(x) and x > 0 else np.nan)
    df["cuota_justa_over25"] = df["p_fair_over25"].apply(lambda x: (1/x) if pd.notna(x) and x > 0 else np.nan)

    # Value% (OVER)
    df["value_pct_model_over25"] = np.where(
        df["cuota_real_over25"].notna() & df["cuota_model_over25"].notna(),
        (df["cuota_real_over25"] - df["cuota_model_over25"]) / df["cuota_model_over25"] * 100, np.nan
    )
    df["value_pct_justa_over25"] = np.where(
        df["cuota_real_over25"].notna() & df["cuota_justa_over25"].notna(),
        (df["cuota_real_over25"] - df["cuota_justa_over25"]) / df["cuota_justa_over25"] * 100, np.nan
    )

    # ===== PEX (OVER) =====
    df["pex_hit_over25"]     = df["p_model_over25"]
    df["pex_ev_norm_over25"] = [
        _pex_ev_norm(p, o) for p, o in zip(df["p_model_over25"], df["cuota_real_over25"])
    ]

    # ===== Lado UNDER 2.5 (derivado) — opcional pero recomendado =====
    df["p_model_under25"] = np.where(df["p_model_over25"].notna(), 1 - df["p_model_over25"], np.nan)
    df["p_fair_under25"]  = np.where(df["p_fair_over25"].notna(), 1 - df["p_fair_over25"], np.nan)

    df["cuota_real_under25"]  = pd.to_numeric(df.get("odds_under25"), errors="coerce")
    df["cuota_model_under25"] = df["p_model_under25"].apply(lambda x: (1/x) if pd.notna(x) and x > 0 else np.nan)
    df["cuota_justa_under25"] = df["p_fair_under25"].apply(lambda x: (1/x) if pd.notna(x) and x > 0 else np.nan)

    df["edge_under25"] = np.where(
        df["p_model_under25"].notna() & df["p_fair_under25"].notna(),
        df["p_model_under25"] - df["p_fair_under25"],
        np.nan
    )
    df["ev_under25"] = np.where(
        df["p_model_under25"].notna() & df["cuota_real_under25"].notna(),
        df["p_model_under25"] * df["cuota_real_under25"] - 1,
        np.nan
    )

    df["value_pct_model_under25"] = np.where(
        df["cuota_real_under25"].notna() & df["cuota_model_under25"].notna(),
        (df["cuota_real_under25"] - df["cuota_model_under25"]) / df["cuota_model_under25"] * 100, np.nan
    )
    df["value_pct_justa_under25"] = np.where(
        df["cuota_real_under25"].notna() & df["cuota_justa_under25"].notna(),
        (df["cuota_real_under25"] - df["cuota_justa_under25"]) / df["cuota_justa_under25"] * 100, np.nan
    )

    # PEX (UNDER)
    df["pex_hit_under25"]     = df["p_model_under25"]
    df["pex_ev_norm_under25"] = [
        _pex_ev_norm(p, o) for p, o in zip(df["p_model_under25"], df["cuota_real_under25"])
    ]

    # Redondeos amables
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].round(3)

    return df
