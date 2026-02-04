from typing import Dict, Any
import argparse
import pandas as pd
import numpy as np
from app.ingest.listado_ligas import enrich_liga_cols
from app.ingest.today_match import get_todays_matches_normalized
from app.ingest.normalizacion.normalizacion_btts import btts_to_df
from app.ingest.filtros import filters
from app.ingest.historico.btts_contexto import enriquecer_btts
from app.ingest.resultados.estado import anotar_estado


def _format_decimal(value, decimals):
    if pd.isna(value):
        return ""
    s = f"{float(value):.{decimals}f}"
    s = s.rstrip("0").rstrip(".")
    return s or "0"


def _format_common(df: pd.DataFrame) -> pd.DataFrame:
    if "ODDS" in df.columns:
        df["ODDS"] = pd.to_numeric(df["ODDS"], errors="coerce").apply(lambda x: _format_decimal(x, 2))
    if "Tarjetas" in df.columns:
        df["Tarjetas"] = pd.to_numeric(df["Tarjetas"], errors="coerce").apply(lambda x: _format_decimal(x, 1))
    if "ROI_estimado" in df.columns:
        df["ROI_estimado"] = pd.to_numeric(df["ROI_estimado"], errors="coerce").apply(lambda x: _format_decimal(x, 1))
    if "Ventaja_modelo" in df.columns:
        df["Ventaja_modelo"] = pd.to_numeric(df["Ventaja_modelo"], errors="coerce").apply(lambda x: _format_decimal(x, 1))
    return df


# ----------------------- Helpers -----------------------
def _apply_optional_threshold(mask: pd.Series, series: pd.Series, thr: Any, op: str = ">=") -> pd.Series:
    """
    Aplica un umbral opcional si 'thr' no es None.
    op: ">=" o "<=".
    """
    if thr is None:
        return mask
    if op == ">=":
        return mask & (series.fillna(-1) >= thr)
    elif op == "<=":
        return mask & (series.fillna(1e9) <= thr)
    return mask


def sanitize_sort_keys(df: pd.DataFrame, keys, fallback):
    keys = list(keys or [])
    valid = [k for k in keys if k in df.columns]
    return valid if valid else [k for k in fallback if k in df.columns]


# ----------------------- Filtros -----------------------
def filter_btts_yes(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Filtra candidatos BTTS-Yes con reglas básicas + PEX opcionales."""
    d = df.copy()

    # Reglas duras base
    mask = (
        d["edge"].fillna(-1) >= cfg["min_edge"]
    ) & (
        d["ev_real"].fillna(-1) >= cfg["min_ev"]
    ) & (
        d["cuota_real_yes"].between(cfg["min_odds"], cfg["max_odds"])
    )

    # (Opcionales) Umbrales PEX
    mask = _apply_optional_threshold(mask, d.get("pex_hit_yes", pd.Series(dtype=float)), cfg.get("min_pex_hit_yes"), ">=")
    mask = _apply_optional_threshold(mask, d.get("pex_ev_norm_yes", pd.Series(dtype=float)), cfg.get("min_pex_ev_yes"), ">=")

    # Señal histórica (opcional si viene btts_potential)
    if "btts_potential" in d.columns:
        mask = mask & (d["btts_potential"].fillna(0) >= cfg.get("min_potential_yes", 0))

    primary = d.loc[mask].copy()
    primary["Rescate"] = False

    fallback_mask = (~mask) & (
        d.get("flag_romper_racha_local", pd.Series(False, index=d.index)).fillna(False) |
        d.get("flag_romper_racha_visita", pd.Series(False, index=d.index)).fillna(False)
    )
    fallback = d.loc[fallback_mask].copy()
    fallback["Rescate"] = True

    out = pd.concat([primary, fallback], ignore_index=True)
    if not out.empty:
        out = out.sort_values(cfg["sort_by"], ascending=False)

    cols = [
        "match_id","home","away","home_id","away_id","competition","country","competition_id","season_id","season_label","kickoff_cdmx",
        "p_model","p_fair_yes","edge","ev_real","odds_btts_yes","btts_potential",
        "cuota_azul","cuota_justa_yes","cuota_real_yes",
        "value_pct_azul","value_pct_justa",
        "local_rate_anota_home","local_rate_encaja_home",
        "visita_rate_anota_away","visita_rate_encaja_away",
        "local_cs_streak","local_cs_max",
        "visita_cs_streak","visita_cs_max",
        "flag_romper_racha_local","flag_romper_racha_visita","Rescate",
        # Nuevas:
        "pex_hit_yes","pex_ev_norm_yes",
        "cards_potential"
    ]
    return out[[c for c in cols if c in out.columns]].reset_index(drop=True)


def filter_btts_no(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Filtra candidatos BTTS-No derivados de los cálculos de Yes + PEX opcionales."""
    d = df.copy()

    # Asegura columnas derivadas (si no las agregaste dentro de btts_to_df)
    if "p_model_no" not in d.columns:
        d["p_model_no"] = np.where(d["p_model"].notna(), 1 - d["p_model"], np.nan)
    if "p_fair_no" not in d.columns:
        d["p_fair_no"] = np.where(d["p_fair_yes"].notna(), 1 - d["p_fair_yes"], np.nan)
    if "cuota_real_no" not in d.columns and "odds_btts_no" in d.columns:
        d["cuota_real_no"] = pd.to_numeric(d["odds_btts_no"], errors="coerce")
    if "cuota_azul_no" not in d.columns:
        d["cuota_azul_no"] = d["p_model_no"].apply(lambda x: (1/x) if (x is not None and x > 0) else np.nan)
    if "cuota_justa_no" not in d.columns:
        d["cuota_justa_no"] = d["p_fair_no"].apply(lambda x: (1/x) if (x is not None and x > 0) else np.nan)
    if "edge_no" not in d.columns:
        d["edge_no"] = np.where(d["p_model_no"].notna() & d["p_fair_no"].notna(),
                                d["p_model_no"] - d["p_fair_no"], np.nan)
    if "ev_no" not in d.columns:
        d["ev_no"] = np.where(d["p_model_no"].notna() & d["cuota_real_no"].notna(),
                              d["p_model_no"] * d["cuota_real_no"] - 1, np.nan)
    if "value_pct_azul_no" not in d.columns:
        d["value_pct_azul_no"] = np.where(
            d["cuota_real_no"].notna() & d["cuota_azul_no"].notna(),
            (d["cuota_real_no"] - d["cuota_azul_no"]) / d["cuota_azul_no"] * 100, np.nan
        )
    if "value_pct_justa_no" not in d.columns:
        d["value_pct_justa_no"] = np.where(
            d["cuota_real_no"].notna() & d["cuota_justa_no"].notna(),
            (d["cuota_real_no"] - d["cuota_justa_no"]) / d["cuota_justa_no"] * 100, np.nan
        )

    # Reglas duras base
    mask = (
        d["edge_no"].fillna(-1) >= cfg["min_edge"]
    ) & (
        d["ev_no"].fillna(-1) >= cfg["min_ev"]
    ) & (
        d["cuota_real_no"].between(cfg["min_odds"], cfg["max_odds"])
    )

    # (Opcionales) Umbrales PEX
    mask = _apply_optional_threshold(mask, d.get("pex_hit_no", pd.Series(dtype=float)), cfg.get("min_pex_hit_no"), ">=")
    mask = _apply_optional_threshold(mask, d.get("pex_ev_norm_no", pd.Series(dtype=float)), cfg.get("min_pex_ev_no"), ">=")

    # Señal histórica inversa: si hay potencial bajo, favorece el NO
    if "btts_potential" in d.columns:
        mask = mask & (d["btts_potential"].fillna(100) <= cfg.get("max_potential_no", 100))

    out = d.loc[mask].sort_values(cfg["sort_by_no"], ascending=False)

    cols = [
        "match_id","home","away","competition","country","competition_id","kickoff_cdmx",
        "p_model_no","p_fair_no","edge_no","ev_no","odds_btts_no","btts_potential",
        "cuota_azul_no","cuota_justa_no","cuota_real_no",
        "value_pct_azul_no","value_pct_justa_no",
        # Nuevas:
        "pex_hit_no","pex_ev_norm_no",
        "cards_potential"
    ]
    return out[[c for c in cols if c in out.columns]].reset_index(drop=True)


# ----------------------- CLI -----------------------
def build_arg_parser(default_cfg: Dict[str, Any], market: str) -> argparse.ArgumentParser:
    """
    Parser que usa como defaults el preset devuelto por `filters(market, n_matches)`.
    Solo expone flags relevantes al mercado.
    """
    p = argparse.ArgumentParser(description=f"Filtra picks {market}")

    # Comunes
    p.add_argument("--min-edge", type=float, default=default_cfg.get("min_edge"))
    p.add_argument("--min-ev", type=float, default=default_cfg.get("min_ev"))
    p.add_argument("--min-odds", type=float, default=default_cfg.get("min_odds"))
    p.add_argument("--max-odds", type=float, default=default_cfg.get("max_odds"))
    p.add_argument("--csv-out", type=str, default=None, help="Ruta CSV para exportar picks")

    # PEX opcionales (si no los usas, omite los flags)
    p.add_argument("--min-pex-hit-yes", type=float, default=default_cfg.get("min_pex_hit_yes"))
    p.add_argument("--min-pex-ev-yes",  type=float, default=default_cfg.get("min_pex_ev_yes"))
    p.add_argument("--min-pex-hit-no",  type=float, default=default_cfg.get("min_pex_hit_no"))
    p.add_argument("--min-pex-ev-no",   type=float, default=default_cfg.get("min_pex_ev_no"))

    # Específicos por mercado
    if market == "BTTS":
        p.add_argument("--min-potential-yes", type=float, default=default_cfg.get("min_potential_yes"))
        p.add_argument("--max-potential-no",  type=float, default=default_cfg.get("max_potential_no"))
    elif market == "OVER25":
        p.add_argument("--min-potential-over", type=float, default=default_cfg.get("min_potential_over"))

    # (Opcional) permitir forzar mercado desde CLI si reutilizas el script
    p.add_argument("--market", type=str, choices=["BTTS", "OVER25"], default=market)

    return p


# ----------------------- Main -----------------------
def main_btts(today_match):
    market = "BTTS"  # Este script es para BTTS; cambia si haces versión OVER25

    # 1) Normaliza y calcula métricas BTTS (incluye pex_* si ya lo añadiste en btts_to_df)
    df_btts = btts_to_df(today_match)
    # 2) Preset desde TU módulo (abierto/moderado/estricto según n_matches)
    preset_cfg: Dict[str, Any] = filters(market, len(df_btts))
    aplicar_filtros = preset_cfg.get("aplicar_filtros", True)

    # 3) Sanea llaves de orden respecto a columnas disponibles (agrego PEX como fallback también)
    sort_by    = sanitize_sort_keys(df_btts, preset_cfg.get("sort_by"),
                                    ["pex_ev_norm_yes", "ev_real", "edge"])
    sort_by_no = sanitize_sort_keys(df_btts, preset_cfg.get("sort_by_no"),
                                    ["pex_ev_norm_no", "ev_no", "edge_no"])

    # 4) CLI con defaults del preset saneado
    preset_cfg_local = {
        **preset_cfg,
        "sort_by": sort_by,
        "sort_by_no": sort_by_no
    }
    args = build_arg_parser(preset_cfg_local, market).parse_args()

    # 5) Construye cfg final (CLI > preset)
    cfg = {
        "min_edge": args.min_edge,
        "min_ev": args.min_ev,
        "min_odds": args.min_odds,
        "max_odds": args.max_odds,
        "min_potential_yes": getattr(args, "min_potential_yes", None),
        "max_potential_no": getattr(args, "max_potential_no", None),
        "min_pex_hit_yes": args.min_pex_hit_yes,
        "min_pex_ev_yes":  args.min_pex_ev_yes,
        "min_pex_hit_no":  args.min_pex_hit_no,
        "min_pex_ev_no":   args.min_pex_ev_no,
        "sort_by": sort_by,
        "sort_by_no": sort_by_no,
    }

    return build_btts_picks_df(cfg, today_match, aplicar_filtros=aplicar_filtros)


def build_btts_picks_df(cfg: Dict[str, Any], today_match, aplicar_filtros: bool = True) -> pd.DataFrame:
    df_btts = btts_to_df(today_match)
    df_btts_contexto = enriquecer_btts(df_btts)

    if aplicar_filtros:
        picks_yes = filter_btts_yes(df_btts_contexto, cfg)
        if picks_yes.empty:
            return pd.DataFrame(columns=[
                "Match_id","Season_id","Hora","Pais","Liga","Local","Visita",
                "ODDS","Mercado","Potencial","Tarjetas","Estado"
            ])
        df_winners = picks_yes.copy()
    else:
        df_winners = df_btts_contexto.copy()

    df_winners["Mercado"] = "BTTS YES"
    df_winners["ROI_estimado"] = df_winners.get("ev_real")
    df_winners["Ventaja_modelo"] = df_winners.get("edge")
    df_winners["ODDS"] = df_winners.get("odds_btts_yes")
    df_winners["PEX_HIT"] = df_winners.get("pex_hit_yes")
    df_winners["PEX_NORM"] = df_winners.get("pex_ev_norm_yes")
    df_winners["Prob_modelo_raw"] = df_winners.get("p_model")

    if aplicar_filtros and "ROI_estimado" in df_winners.columns:
        df_winners = df_winners[df_winners["ROI_estimado"] >= cfg["min_ev"]].reset_index(drop=True)

    if "Prob_modelo_raw" in df_winners.columns:
        df_winners["Prob_modelo"] = (
            pd.to_numeric(df_winners["Prob_modelo_raw"], errors="coerce") * 100
        ).round(0).astype("Int64")
        df_winners = df_winners.drop(columns=["Prob_modelo_raw"])

    df_winners_final = enrich_liga_cols(df_winners, competition_id_col="competition_id", season_id_col="season_id")

    df_winners_final = df_winners_final.rename(columns={
        "kickoff_cdmx": "Hora",
        "match_id": "Match_id",
        "season_id": "Season_id",
        "season_label": "Season_label",
        "home": "Local",
        "away": "Visita",
        "btts_potential": "Potencial",
        "cards_potential": "Tarjetas",
    })

    if "PEX_HIT" in df_winners_final.columns:
        df_winners_final["PROB_Historico"] = (
            pd.to_numeric(df_winners_final["PEX_HIT"], errors="coerce") * 100
        ).round(0).astype("Int64")

    if "home_id" in df_winners_final.columns and "homeID" not in df_winners_final.columns:
        df_winners_final["homeID"] = df_winners_final["home_id"]
    if "away_id" in df_winners_final.columns and "awayID" not in df_winners_final.columns:
        df_winners_final["awayID"] = df_winners_final["away_id"]

    df_winners_final["competition_id"] = df_winners.get("competition_id")

    cols_base = [
        "Match_id","Season_id","Season_label","competition_id","home_id","away_id","homeID","awayID", "Hora", "Pais", "Liga", "Local", "Visita",
        "ODDS", "Mercado", "Potencial","Prob_modelo","Tarjetas",
        "ROI_estimado", "Ventaja_modelo",
        "PROB_Historico"
    ]
    cols_base = [c for c in cols_base if c in df_winners_final.columns]
    if "Hora" in cols_base:
        df_final = df_winners_final[cols_base].sort_values("Hora")
    else:
        df_final = df_winners_final[cols_base]
    df_final = anotar_estado(df_final, "BTTS")
    desired_cols = [
        "Match_id","Season_id","Season_label","competition_id","home_id","away_id","homeID","awayID", "Hora", "Pais", "Liga", "Local", "Visita",
        "ODDS", "Mercado", "Marcador","Potencial","Prob_modelo","Tarjetas","Estado",
        "ROI_estimado", "Ventaja_modelo","PROB_Historico"
    ]
    df_final = df_final[[c for c in desired_cols if c in df_final.columns]]
    if "ROI_estimado" in df_final.columns:
        df_final["ROI_estimado"] = (
            pd.to_numeric(df_final["ROI_estimado"], errors="coerce") * 100
        )
    if "Ventaja_modelo" in df_final.columns:
        df_final["Ventaja_modelo"] = (
            pd.to_numeric(df_final["Ventaja_modelo"], errors="coerce") * 100
        )
    df_final = _format_common(df_final)
    if "PEX_NORM" in df_final.columns:
        df_final = df_final.drop(columns=["PEX_NORM"])
    return df_final

def _format_column(df: pd.DataFrame, column: str, fmt: str) -> None:
    if column in df.columns:
        df[column] = df[column].apply(
            lambda x: "" if pd.isna(x) else format(x, fmt)
        )


def _format_common(df: pd.DataFrame) -> pd.DataFrame:
    _format_column(df, "ODDS", ".2f")
    _format_column(df, "Tarjetas", ".1f")
    _format_column(df, "ROI_estimado", ".1f")
    _format_column(df, "Ventaja_modelo", ".1f")
    return df
