from __future__ import annotations

import argparse
from datetime import datetime
from typing import Any, Dict

import numpy as np
import pandas as pd

from app.ingest.filtros import filters
from app.ingest.listado_ligas import enrich_liga_cols
from app.ingest.normalizacion.normalizacion_corners import corners_to_df
from app.ingest.resultados.estado import anotar_estado
from app.ingest.historico.corners_contexto import enriquecer_corners
from app.ingest.modelos.corners_under95_model import aplicar_modelo as modelo_corners


def build_arg_parser(default_cfg: Dict[str, Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Córners Under 9.5")
    parser.add_argument("--max-potential-ft", type=float, default=default_cfg.get("max_potential_ft"))
    parser.add_argument("--max-potential-o95", type=float, default=default_cfg.get("max_o95_potential"))
    parser.add_argument("--min-odds-under95", type=float, default=default_cfg.get("min_odds_under95"))
    parser.add_argument("--max-odds-under95", type=float, default=default_cfg.get("max_odds_under95"))
    return parser


def _merge_league_meta(df: pd.DataFrame) -> pd.DataFrame:
    return enrich_liga_cols(df, competition_id_col="competition_id", season_id_col="season_id")


def _format_output(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    float_cols = ["Potencial_total", "Odds_U95", "EV_modelo"]
    int_cols = ["Corners_local", "Corners_visitante", "Corners_total",
                "Potencial_u95", "Prob_modelo", "PROB_Historico"]
    for col in float_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(2)
    for col in int_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)
    return out


def _apply_under_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    totals = pd.to_numeric(out.get("Corners_total"), errors="coerce")
    registro_flag = out.get("Tiene_registro")
    kickoff = pd.to_datetime(out.get("Hora"), errors="coerce")
    now = datetime.utcnow()
    zero_corners = (
        pd.to_numeric(out.get("Corners_local"), errors="coerce").fillna(0) == 0
    ) & (
        pd.to_numeric(out.get("Corners_visitante"), errors="coerce").fillna(0) == 0
    )
    if registro_flag is not None:
        # Si hay registro oficial úsalo, pero permite totales calculados para marcar el resultado.
        mask = registro_flag.fillna(False) | totals.notna()
    else:
        mask = totals.notna()
    if kickoff is not None:
        mask &= kickoff.notna() & (kickoff <= now)
    under_mask = mask & (totals <= 9)
    over_mask = mask & (totals > 9)
    pre_match_mask = zero_corners & kickoff.notna() & (kickoff > now) & (~mask)
    live_mask = zero_corners & kickoff.notna() & (kickoff <= now) & (~mask)
    out.loc[under_mask, "Estado"] = "VERDE"
    out.loc[under_mask, "Acierto"] = "Acierto"
    out.loc[over_mask, "Estado"] = "ROJO"
    out.loc[over_mask, "Acierto"] = "Fallo"
    out.loc[pre_match_mask, "Estado"] = "Pendiente"
    out.loc[pre_match_mask, "Acierto"] = ""
    out.loc[live_mask, "Estado"] = "LIVE"
    out.loc[live_mask, "Acierto"] = ""
    out["Estado"] = out["Estado"].fillna("")
    out["Acierto"] = out["Acierto"].fillna("")
    return out


def build_corners_under95_df(base: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    if base.empty:
        return pd.DataFrame(columns=[
            "Partido", "Season_id", "Hora", "Pais", "Liga",
            "Local", "Visitante", "Mercado",
            "Corners_local", "Corners_visitante", "Corners_total",
            "Potencial_total", "Potencial_u95", "Odds_U95",
            "Estado", "Acierto", "PROB_Historico"
        ])

    base = enriquecer_corners(base)
    enriched = _merge_league_meta(base).rename(columns={
        "match_id": "Partido",
        "season_id": "Season_id",
        "home": "Local",
        "away": "Visitante",
        "kickoff_local_cdmx": "Hora",
    })
    ft_df = enriched.copy()
    ft_df["Mercado"] = "Under 9.5 corners"
    # Toma las cuentas finales por equipo directamente del endpoint.
    ft_df["Corners_local"] = ft_df.get("team_a_corners")
    ft_df["Corners_visitante"] = ft_df.get("team_b_corners")
    ft_df["Tiene_registro"] = ft_df["ft_total"].notna()
    # Solo total real; no usar potencial como valor final para resultados.
    ft_df["Corners_total"] = ft_df["ft_total"]
    ft_df["Potencial_total"] = ft_df["corners_potential"]
    ft_df["Potencial_u95"] = 100 - ft_df["corners_o95_potential"].fillna(0)
    ft_df["Odds_U95"] = ft_df["odds_corners_under_95"]

    max_potential_total = cfg.get("max_potential_ft", 10)
    max_potential_o95 = cfg.get("max_o95_potential", 55)
    mask_total = ft_df["Potencial_total"].fillna(999) <= max_potential_total
    mask_o95 = ft_df["corners_o95_potential"].fillna(100) <= max_potential_o95
    strict_mask = mask_total & mask_o95

    ft_df = modelo_corners(ft_df.copy())
    if "Prob_modelo" in ft_df.columns:
        ft_df["Prob_modelo"] = (
            pd.to_numeric(ft_df["Prob_modelo"], errors="coerce") * 100
        ).round(0).astype("Int64")

    filtered = ft_df.loc[strict_mask].copy()
    filtered["Rescate"] = False
    filtered["PROB_Historico"] = filtered["Potencial_u95"]

    keep_cols = [
        "Partido", "Season_id", "Hora", "Pais", "Liga", "Local", "Visitante",
        "Mercado", "Odds_U95",
        "Corners_local", "Corners_visitante", "Corners_total",
        "Potencial_total", "Potencial_u95",
        "Prob_modelo", "EV_modelo",
        "Estado", "Acierto", "PROB_Historico", "Tiene_registro", "Rescate"
    ]
    for col in keep_cols:
        if col not in filtered.columns:
            filtered[col] = np.nan
    filtered = filtered[keep_cols]
    filtered = filtered.sort_values(["Hora", "Partido"], na_position="last")
    filtered = anotar_estado(filtered.reset_index(drop=True), "CORNERS")
    filtered = _apply_under_outcomes(filtered)
    if "Tiene_registro" in filtered.columns:
        filtered = filtered.drop(columns=["Tiene_registro"])
    return _format_output(filtered)


def main_corners_u95(today_match):
    market = "CORNERS_U95"
    base = corners_to_df(today_match).copy()
    preset_cfg: Dict[str, Any] = filters(market, len(base))
    args = build_arg_parser(preset_cfg).parse_args()
    cfg = {
        "max_potential_ft": args.max_potential_ft,
        "max_o95_potential": args.max_potential_o95,
        "min_odds_under95": args.min_odds_under95,
        "max_odds_under95": args.max_odds_under95,
    }
    df = build_corners_under95_df(base, cfg)
    min_odds = cfg.get("min_odds_under95", 1.0)
    max_odds = cfg.get("max_odds_under95", 99.0)
    if not df.empty and "Odds_U95" in df.columns:
        odds = pd.to_numeric(df["Odds_U95"], errors="coerce")
        df = df.loc[odds.between(min_odds, max_odds, inclusive="both")].reset_index(drop=True)
    return df
