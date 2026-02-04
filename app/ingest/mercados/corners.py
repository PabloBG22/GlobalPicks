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
from app.ingest.modelos.corners_over85_model import aplicar_modelo as modelo_corners


def build_arg_parser(default_cfg: Dict[str, Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Córners HT/FT")
    parser.add_argument("--min-corners-ht", type=float, default=default_cfg.get("min_corners_ht"))
    parser.add_argument("--min-corners-ft", type=float, default=default_cfg.get("min_corners_ft"))
    parser.add_argument("--min-potential-ft", type=float, default=default_cfg.get("min_potential_ft"))
    parser.add_argument("--min-potential-o85", type=float, default=default_cfg.get("min_o85_potential"))
    parser.add_argument("--min-odds-over85", type=float, default=default_cfg.get("min_odds_over85"))
    parser.add_argument("--max-odds-over85", type=float, default=default_cfg.get("max_odds_over85"))
    return parser


def _merge_league_meta(df: pd.DataFrame) -> pd.DataFrame:
    return enrich_liga_cols(df, competition_id_col="competition_id", season_id_col="season_id")


def _format_output(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    float_cols = ["Potencial_total", "ODDS", "EV_modelo"]
    int_cols = ["Corners_local", "Corners_visitante", "Corners_total", "Potencial_o85", "PROB_Historico", "Prob_modelo"]

    for col in float_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(2)
    for col in int_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)
    return out


def _apply_corner_outcomes(df: pd.DataFrame) -> pd.DataFrame:
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
        # Usa registro oficial cuando está disponible, pero permite totales calculados para marcar resultado.
        mask = registro_flag.fillna(False) | totals.notna()
    else:
        mask = totals.notna()
    if kickoff is not None:
        mask &= kickoff.notna() & (kickoff <= now)

    pre_match_mask = zero_corners & kickoff.notna() & (kickoff > now) & (~mask)
    live_mask = zero_corners & kickoff.notna() & (kickoff <= now) & (~mask)

    over_mask = mask & (totals >= 9)
    under_mask = mask & (totals < 9)

    if "Estado" not in out.columns:
        out["Estado"] = ""
    if "Acierto" not in out.columns:
        out["Acierto"] = ""

    out.loc[pre_match_mask, "Estado"] = "Pendiente"
    out.loc[pre_match_mask, "Acierto"] = ""
    out.loc[live_mask, "Estado"] = "LIVE"
    out.loc[live_mask, "Acierto"] = ""

    out.loc[over_mask, "Estado"] = "VERDE"
    out.loc[over_mask, "Acierto"] = "Acierto"
    out.loc[under_mask, "Estado"] = "ROJO"
    out.loc[under_mask, "Acierto"] = "Fallo"

    out["Estado"] = out["Estado"].fillna("")
    out["Acierto"] = out["Acierto"].fillna("")
    return out


def build_corners_market_df(base: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    if base.empty:
        return pd.DataFrame(columns=[
            "Partido", "Season_id", "Hora", "Pais", "Liga", "Local", "Visitante",
            "Mercado", "Corners_local", "Corners_visitante", "Corners_total",
            "Potencial_total", "Potencial_o85",
            "ODDS", "Estado", "Acierto", "PROB_Historico"
        ])

    base = enriquecer_corners(base)
    enriched = _merge_league_meta(base)
    enriched = enriched.rename(columns={
        "match_id": "Partido",
        "season_id": "Season_id",
        "home": "Local",
        "away": "Visitante",
        "kickoff_local_cdmx": "Hora",
    })
    if "home_id" in enriched.columns and "homeID" not in enriched.columns:
        enriched["homeID"] = enriched["home_id"]
    if "away_id" in enriched.columns and "awayID" not in enriched.columns:
        enriched["awayID"] = enriched["away_id"]

    ft_df = enriched.copy()
    ft_df["Mercado"] = "Over 8.5 corners"
    # Usa directamente los contadores de córners finales por equipo desde league-matches.
    ft_df["Corners_local"] = ft_df.get("team_a_corners")
    ft_df["Corners_visitante"] = ft_df.get("team_b_corners")
    ft_df["Tiene_registro"] = ft_df["ft_total"].notna()
    # Usa solo el total real; no rellenar con potencial para no contaminar resultados.
    ft_df["Corners_total"] = ft_df["ft_total"]
    ft_df["Potencial_total"] = ft_df["corners_potential"]
    ft_df["Potencial_o85"] = ft_df["corners_o85_potential"]
    ft_df["ODDS"] = ft_df["odds_corners_over_85"]
    ft_df = ft_df.rename(columns={
        "local_corners_over85_rate": "Local_corners_rate",
        "visita_corners_over85_rate": "Visita_corners_rate",
        "local_under85_streak": "Local_under85_streak",
        "visita_under85_streak": "Visita_under85_streak",
    })

    min_potential_total = cfg.get("min_potential_ft", 10)
    min_potential_o85 = cfg.get("min_o85_potential", 70)
    racha_rate = cfg.get("racha_min_corners_rate", 0.7)
    mask_total = ft_df["Potencial_total"].fillna(0) >= min_potential_total
    mask_o85 = ft_df["Potencial_o85"].fillna(0) >= min_potential_o85
    strict_mask = mask_total & mask_o85
    ft_df = modelo_corners(ft_df.copy())
    if "Prob_modelo" in ft_df.columns:
        ft_df["Prob_modelo"] = (
            pd.to_numeric(ft_df["Prob_modelo"], errors="coerce") * 100
        ).round(0).astype("Int64")

    strict_df = ft_df.loc[strict_mask].copy()
    strict_df["Rescate"] = False

    rescue = pd.DataFrame()
    if {"Local_corners_rate","Visita_corners_rate"} <= set(ft_df.columns):
        home_signal = (
            ft_df["Local_corners_rate"].fillna(0) >= racha_rate
        ) & (ft_df.get("Local_under85_streak", pd.Series(0, index=ft_df.index)).fillna(0) >= 1)
        away_signal = (
            ft_df["Visita_corners_rate"].fillna(0) >= racha_rate
        ) & (ft_df.get("Visita_under85_streak", pd.Series(0, index=ft_df.index)).fillna(0) >= 1)
        rescue_mask = (~strict_mask) & (home_signal | away_signal)
        rescue = ft_df.loc[rescue_mask].copy()
        if not rescue.empty:
            rescue["Rescate"] = True

    combined = pd.concat([strict_df, rescue], ignore_index=True) if not rescue.empty else strict_df.copy()

    combined["Season_label"] = enriched.get("season_label")
    combined["competition_id"] = enriched.get("competition_id")
    combined = combined.rename(columns={"Odds_O85": "ODDS"})

    keep_cols = [
        "Partido", "Season_id","Season_label","competition_id","home_id","away_id","homeID","awayID", "Hora", "Pais", "Liga", "Local", "Visitante",
        "Mercado", "ODDS", "Corners_local", "Corners_visitante", "Corners_total",
        "Potencial_total", "Potencial_o85", "Prob_modelo", "EV_modelo",
        "Estado", "Acierto", "PROB_Historico", "Tiene_registro","Rescate"
    ]
    for col in keep_cols:
        if col not in combined.columns:
            combined[col] = np.nan

    combined = combined[keep_cols]
    combined = combined.sort_values(["Hora", "Partido"], na_position="last")
    combined = anotar_estado(combined.reset_index(drop=True), "CORNERS")
    combined = _apply_corner_outcomes(combined)
    combined["PROB_Historico"] = pd.to_numeric(combined.get("Potencial_o85"), errors="coerce")
    if "Tiene_registro" in combined.columns:
        combined = combined.drop(columns=["Tiene_registro"])
    return _format_output(combined)


def main_corners(today_match):
    market = "CORNERS"
    base = corners_to_df(today_match).copy()
    preset_cfg: Dict[str, Any] = filters(market, len(base))
    args = build_arg_parser(preset_cfg).parse_args()
    cfg = {
        "min_corners_ht": args.min_corners_ht,
        "min_corners_ft": args.min_corners_ft,
        "min_potential_ft": args.min_potential_ft,
        "min_o85_potential": args.min_potential_o85,
        "min_odds_over85": args.min_odds_over85,
        "max_odds_over85": args.max_odds_over85,
    }

    df = build_corners_market_df(base, cfg)
    return df
