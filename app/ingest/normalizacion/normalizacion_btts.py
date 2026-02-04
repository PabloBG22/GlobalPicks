import pandas as pd
import numpy as np
from math import exp
from typing import Any, Dict, List, Optional
from app.ingest.normalizacion.two_way import fair_prob_two_way as two_way

def poisson_btts(lam_a: float, lam_b: float) -> float:
    """P(BTTS) = 1 - e^-λa - e^-λb + e^-(λa+λb)."""
    return 1.0 - exp(-lam_a) - exp(-lam_b) + exp(-(lam_a + lam_b))

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

# ---------------- BTTS ----------------
def btts_to_df(normalized: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for n in normalized:
        lam_a = float(n.get("team_a_xg_prematch") or n.get("home_xg_prematch") or 0.0)
        lam_b = float(n.get("team_b_xg_prematch") or n.get("away_xg_prematch") or 0.0)

        odds_yes = n.get("odds_btts_yes")
        odds_no  = n.get("odds_btts_no")

        p_model = poisson_btts(lam_a, lam_b) if (lam_a > 0 and lam_b > 0) else None
        p_fair_yes = two_way(odds_yes, odds_no)

        edge = (p_model - p_fair_yes) if (p_model is not None and p_fair_yes is not None) else None
        ev_real = (p_model * float(odds_yes) - 1) if (p_model is not None and odds_yes) else None

        # Robust: acepta varias formas y objetos anidados
        comp = (
            n.get("competition_name")
            or (n.get("competition") or {}).get("name")
            or n.get("league_name")
            or n.get("competition")
            or n.get("league")
            or ""
        )
        country = (
            n.get("country")
            or (n.get("competition") or {}).get("country")
            or n.get("country_name")
            or n.get("nation")
            or ""
        )

        # Limpieza
        comp = comp.strip() if isinstance(comp, str) else ""
        country = country.strip() if isinstance(country, str) else ""

        rows.append({
            "match_id" : n.get("match_id"),
            "season_id": n.get("season_id"),
            "season_label": n.get("season_label"),
            "home": n.get("home"),
            "away": n.get("away"),
            "home_id": n.get("home_id"),
            "away_id": n.get("away_id"),
            "competition_id": n.get("competition_id"),
            "competition": comp,
            "country": country,
            "kickoff_cdmx": n.get("kickoff_local_cdmx"),
            "odds_btts_yes": odds_yes,
            "odds_btts_no": odds_no,
            "p_model": round(p_model, 4) if p_model is not None else None,
            "p_fair_yes": round(p_fair_yes, 4) if p_fair_yes is not None else None,
            "edge": round(edge, 4) if edge is not None else None,
            "ev_real": round(ev_real, 4) if ev_real is not None else None,
            "btts_potential": n.get("btts_potential"),
            "cards_potential": n.get("cards_potential"),
        })

    df = pd.DataFrame(rows)

    # ====== Cuotas (BTTS YES) ======
    df["cuota_azul"]      = df["p_model"].apply(lambda x: (1/x) if (x is not None and x > 0) else np.nan)
    df["cuota_justa_yes"] = df["p_fair_yes"].apply(lambda x: (1/x) if (x is not None and x > 0) else np.nan)
    df["cuota_real_yes"]  = pd.to_numeric(df["odds_btts_yes"], errors="coerce")

    # ====== Value% (BTTS YES) ======
    df["value_pct_azul"] = np.where(
        df["cuota_real_yes"].notna() & df["cuota_azul"].notna(),
        (df["cuota_real_yes"] - df["cuota_azul"]) / df["cuota_azul"] * 100.0,
        np.nan
    )
    df["value_pct_justa"] = np.where(
        df["cuota_real_yes"].notna() & df["cuota_justa_yes"].notna(),
        (df["cuota_real_yes"] - df["cuota_justa_yes"]) / df["cuota_justa_yes"] * 100.0,
        np.nan
    )

    # Redondeos amigables (BTTS YES)
    for col in ["cuota_azul", "cuota_justa_yes", "cuota_real_yes", "value_pct_azul", "value_pct_justa"]:
        df[col] = df[col].round(3)

    # ====== BTTS NO (derivado del YES) ======
    df["p_model_no"] = np.where(df["p_model"].notna(), 1 - df["p_model"], np.nan)
    df["p_fair_no"]  = np.where(df["p_fair_yes"].notna(), 1 - df["p_fair_yes"], np.nan)

    df["cuota_real_no"]  = pd.to_numeric(df["odds_btts_no"], errors="coerce")
    df["cuota_azul_no"]  = df["p_model_no"].apply(lambda x: (1/x) if (x is not None and x > 0) else np.nan)
    df["cuota_justa_no"] = df["p_fair_no"].apply(lambda x: (1/x) if (x is not None and x > 0) else np.nan)

    df["edge_no"] = np.where(
        df["p_model_no"].notna() & df["p_fair_no"].notna(),
        df["p_model_no"] - df["p_fair_no"],
        np.nan
    )
    df["ev_no"] = np.where(
        df["p_model_no"].notna() & df["cuota_real_no"].notna(),
        df["p_model_no"] * df["cuota_real_no"] - 1,
        np.nan
    )

    df["value_pct_azul_no"] = np.where(
        df["cuota_real_no"].notna() & df["cuota_azul_no"].notna(),
        (df["cuota_real_no"] - df["cuota_azul_no"]) / df["cuota_azul_no"] * 100,
        np.nan
    )
    df["value_pct_justa_no"] = np.where(
        df["cuota_real_no"].notna() & df["cuota_justa_no"].notna(),
        (df["cuota_real_no"] - df["cuota_justa_no"]) / df["cuota_justa_no"] * 100,
        np.nan
    )

    # Redondeo columnas extra (BTTS NO)
    cols_no = [
        "cuota_azul_no","cuota_justa_no","cuota_real_no",
        "edge_no","ev_no","value_pct_azul_no","value_pct_justa_no"
    ]
    df[cols_no] = df[cols_no].round(3)

    # ====== PEX ======
    # Probabilidad de acierto (hit)
    df["pex_hit_yes"] = df["p_model"]
    df["pex_hit_no"]  = df["p_model_no"]

    # EV normalizado a [0,1] usando la cuota real
    df["pex_ev_norm_yes"] = [
        _pex_ev_norm(p, o) for p, o in zip(df["p_model"], df["cuota_real_yes"])
    ]
    df["pex_ev_norm_no"] = [
        _pex_ev_norm(p, o) for p, o in zip(df["p_model_no"], df["cuota_real_no"])
    ]

    # Redondeo PEX
    for col in ["pex_hit_yes", "pex_hit_no", "pex_ev_norm_yes", "pex_ev_norm_no"]:
        df[col] = df[col].round(3)

    return df
