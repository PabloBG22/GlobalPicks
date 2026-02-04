# app/ingest/normalizacion/normalizacion_o05ht.py
import math
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from app.ingest.normalizacion.two_way import fair_prob_two_way as two_way

HT_SHARE_DEFAULT = 0.45

def _poisson_o05ht_from_sum(lam_ht: float) -> Optional[float]:
    if lam_ht is None or lam_ht <= 0:
        return None
    return 1.0 - math.exp(-lam_ht)

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

def _safe(x, default: float = 0.0) -> float:
    if x is None:
        return default
    try:
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return default
        return v
    except Exception:
        return default

def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))

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

def o05ht_to_df(
    normalized: List[Dict[str, Any]],
    ht_share: float = HT_SHARE_DEFAULT
) -> pd.DataFrame:

    # --- Pesos por defecto para el PEX HT (se conserva) ---
    DEFAULT_WEIGHTS_HT: Dict[str, float] = {
        "p_model_ht": 0.40,
        "o05HT_potential": 0.20,
        "o15HT_potential": 0.10,
        "btts_fhg": 0.05,
        "xg_ht": 0.10,
        "ht_share": 0.05,
        "edge": 0.05,
        "ev": 0.05,
    }

    # --- Bandas de PEX (opcional, se conserva) ---
    DEFAULT_BANDS = {
        "estricto": {"pex": 0.75, "ev": 0.08},
        "moderado": {"pex": 0.65, "ev": 0.05},
        "abierto":  {"pex": 0.55, "ev": 0.00},
    }

    def _compute_pex_row(row: pd.Series, w: Dict[str, float]) -> Optional[float]:
        """
        PEX HT en [0,1] usando ponderación de señales (tu versión original, conservada).
        """
        total_w = 0.0
        score = 0.0

        score += _clip01(_safe(row.get("p_model_o05HT"))) * w.get("p_model_ht", 0.0); total_w += w.get("p_model_ht", 0.0)
        score += _clip01(_safe(row.get("o05HT_potential"))/100.0) * w.get("o05HT_potential", 0.0); total_w += w.get("o05HT_potential", 0.0)
        score += _clip01(_safe(row.get("o15HT_potential"))/100.0) * w.get("o15HT_potential", 0.0); total_w += w.get("o15HT_potential", 0.0)
        score += _clip01(_safe(row.get("btts_fhg_potential"))/100.0) * w.get("btts_fhg", 0.0); total_w += w.get("btts_fhg", 0.0)

        xg_ht_norm = _clip01(_safe(row.get("xg_ht_aprox")) / 2.0)
        score += xg_ht_norm * w.get("xg_ht", 0.0); total_w += w.get("xg_ht", 0.0)

        score += _clip01(_safe(row.get("ht_share"))) * w.get("ht_share", 0.0); total_w += w.get("ht_share", 0.0)

        edge_pos = max(0.0, _safe(row.get("edge_o05HT"))); score += edge_pos * w.get("edge", 0.0); total_w += w.get("edge", 0.0)
        ev_pos   = max(0.0, _safe(row.get("ev_o05HT")));   score += ev_pos   * w.get("ev", 0.0);   total_w += w.get("ev", 0.0)

        if total_w <= 0:
            return None
        return round(score / total_w, 3)

    def _band_row(row: pd.Series, bands: Dict[str, Dict[str, float]]) -> str:
        pex = _safe(row.get("pex_ht"))
        ev  = _safe(row.get("ev_o05HT"))
        if pex >= bands["estricto"]["pex"] and ev >= bands["estricto"]["ev"]:
            return "estricto"
        if pex >= bands["moderado"]["pex"] and ev >= bands["moderado"]["ev"]:
            return "moderado"
        if pex >= bands["abierto"]["pex"] and ev >= bands["abierto"]["ev"]:
            return "abierto"
        return "fuera"

    rows = []
    for n in normalized:
        lam_a = float(n.get("team_a_xg_prematch") or n.get("home_xg_prematch") or 0.0)
        lam_b = float(n.get("team_b_xg_prematch") or n.get("away_xg_prematch") or 0.0)

        total_xg_pre = n.get("total_xg_prematch")
        try:
            total_xg_pre = float(total_xg_pre) if total_xg_pre is not None else None
        except (TypeError, ValueError):
            total_xg_pre = None

        lam_sum = total_xg_pre if (total_xg_pre and total_xg_pre > 0) else ((lam_a + lam_b) if (lam_a > 0 and lam_b > 0) else None)
        lam_ht = (lam_sum * ht_share) if lam_sum is not None else None

        # ---------- Cuotas HT con alias robustos ----------
        odds_over_ht = _first_float(n, [
            "odds_1st_half_over05", "odds_first_half_over_05",
            "odds_over05_ht", "odds_over_05_ht", "odds_1h_over05",
            "odds_over05_1h"
        ])
        odds_under_ht = _first_float(n, [
            "odds_1st_half_under05", "odds_first_half_under_05",
            "odds_under05_ht", "odds_under_05_ht", "odds_1h_under05",
            "odds_under05_1h"
        ])

        # ---------- Modelo / Fair / EV ----------
        p_model_o05ht = _poisson_o05ht_from_sum(lam_ht) if lam_ht is not None else None
        p_fair_o05ht  = two_way(odds_over_ht, odds_under_ht) if (odds_over_ht and odds_under_ht) else None
        edge_o05ht    = (p_model_o05ht - p_fair_o05ht) if (p_model_o05ht is not None and p_fair_o05ht is not None) else None
        ev_o05ht      = (p_model_o05ht * odds_over_ht - 1) if (p_model_o05ht is not None and odds_over_ht) else None

        # ---------- Potencial ----------
        o05HT_potential = n.get("o05HT_potential")
        if o05HT_potential is None:
            o05HT_potential = round(p_model_o05ht * 100, 2) if p_model_o05ht is not None else None

        rows.append({
            "match_id": n.get("match_id"),
            "home_id": n.get("home_id") or n.get("homeID"),
            "away_id": n.get("away_id") or n.get("awayID"),
            "home": n.get("home"),
            "away": n.get("away"),
            "competition_id": n.get("competition_id"),
            "kickoff_cdmx": n.get("kickoff_local_cdmx"),

            # cuotas
            "odds_o05HT": odds_over_ht,
            "odds_u05HT": odds_under_ht,   # <- útil para debug / fair

            # métricas Over 0.5 HT
            "p_model_o05HT": round(p_model_o05ht, 4) if p_model_o05ht is not None else None,
            "p_fair_o05HT": round(p_fair_o05ht, 4) if p_fair_o05ht is not None else None,
            "edge_o05HT": round(edge_o05ht, 4) if edge_o05ht is not None else None,
            "ev_o05HT": round(ev_o05ht, 4) if ev_o05ht is not None else None,

            "o05HT_potential": o05HT_potential,

            # señales extra para filtros
            "o15HT_potential": n.get("o15HT_potential"),
            "btts_fhg_potential": n.get("btts_fhg_potential"),

            # trazabilidad
            "xg_total_prematch": round(lam_sum, 3) if lam_sum is not None else None,
            "xg_ht_aprox": round(lam_ht, 3) if lam_ht is not None else None,
            "ht_share": round(ht_share, 3),
            "cards_potential": n.get("cards_potential"),
        })

    df = pd.DataFrame(rows)

    # ===== Cuotas reales y de referencia (OVER) =====
    df["cuota_real_o05HT"]  = pd.to_numeric(df.get("odds_o05HT"), errors="coerce")
    df["cuota_model_o05HT"] = df["p_model_o05HT"].apply(lambda x: (1/x) if pd.notna(x) and x > 0 else np.nan)
    df["cuota_justa_o05HT"] = df["p_fair_o05HT"].apply(lambda x: (1/x) if pd.notna(x) and x > 0 else np.nan)

    # Value% (OVER)
    df["value_pct_model_o05HT"] = np.where(
        df["cuota_real_o05HT"].notna() & df["cuota_model_o05HT"].notna(),
        (df["cuota_real_o05HT"] - df["cuota_model_o05HT"]) / df["cuota_model_o05HT"] * 100,
        np.nan
    )
    df["value_pct_justa_o05HT"] = np.where(
        df["cuota_real_o05HT"].notna() & df["cuota_justa_o05HT"].notna(),
        (df["cuota_real_o05HT"] - df["cuota_justa_o05HT"]) / df["cuota_justa_o05HT"] * 100,
        np.nan
    )

    # ====== Lado UNDER 0.5 HT (derivado) ======
    df["p_model_u05HT"] = np.where(df["p_model_o05HT"].notna(), 1 - df["p_model_o05HT"], np.nan)
    df["p_fair_u05HT"]  = np.where(df["p_fair_o05HT"].notna(), 1 - df["p_fair_o05HT"], np.nan)

    df["cuota_real_u05HT"]  = pd.to_numeric(df.get("odds_u05HT"), errors="coerce")
    df["cuota_model_u05HT"] = df["p_model_u05HT"].apply(lambda x: (1/x) if pd.notna(x) and x > 0 else np.nan)
    df["cuota_justa_u05HT"] = df["p_fair_u05HT"].apply(lambda x: (1/x) if pd.notna(x) and x > 0 else np.nan)

    df["edge_u05HT"] = np.where(
        df["p_model_u05HT"].notna() & df["p_fair_u05HT"].notna(),
        df["p_model_u05HT"] - df["p_fair_u05HT"],
        np.nan
    )
    df["ev_u05HT"] = np.where(
        df["p_model_u05HT"].notna() & df["cuota_real_u05HT"].notna(),
        df["p_model_u05HT"] * df["cuota_real_u05HT"] - 1,
        np.nan
    )

    df["value_pct_model_u05HT"] = np.where(
        df["cuota_real_u05HT"].notna() & df["cuota_model_u05HT"].notna(),
        (df["cuota_real_u05HT"] - df["cuota_model_u05HT"]) / df["cuota_model_u05HT"] * 100,
        np.nan
    )
    df["value_pct_justa_u05HT"] = np.where(
        df["cuota_real_u05HT"].notna() & df["cuota_justa_u05HT"].notna(),
        (df["cuota_real_u05HT"] - df["cuota_justa_u05HT"]) / df["cuota_justa_u05HT"] * 100,
        np.nan
    )

    # ---------- PEX HT (ponderado, se conserva) ----------
    df["pex_ht"] = df.apply(lambda r: _compute_pex_row(r, DEFAULT_WEIGHTS_HT), axis=1)

    # ---------- Banda PEX (opcional, se conserva) ----------
    df["pex_band"] = df.apply(lambda r: _band_row(r, DEFAULT_BANDS), axis=1)

    # ---------- NUEVO: PEX_HIT & PEX_NORM (Over y Under) ----------
    # Over 0.5 HT
    df["pex_hit_o05HT"]      = df["p_model_o05HT"]
    df["pex_ev_norm_o05HT"]  = [
        _pex_ev_norm(p, o) for p, o in zip(df["p_model_o05HT"], df["cuota_real_o05HT"])
    ]

    # Under 0.5 HT
    df["pex_hit_u05HT"]      = df["p_model_u05HT"]
    df["pex_ev_norm_u05HT"]  = [
        _pex_ev_norm(p, o) for p, o in zip(df["p_model_u05HT"], df["cuota_real_u05HT"])
    ]

    # Redondeos finales suaves
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].round(3)

    return df
