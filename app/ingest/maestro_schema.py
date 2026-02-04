from __future__ import annotations

from typing import Iterable

import pandas as pd


CANONICAL_COLUMNS: list[str] = [
    "Fecha_ejecucion",
    "Hora",
    "Pais",
    "Liga",
    "Local",
    "Visitante",
    "Mercado",
    "market_group",
    "Match_id",
    "competition_id",
    "Season_label",
    "home_id",
    "away_id",
    "ODDS",
    "ODDS_metricas",
    "Telegram_estado",
    "Telegram_message_id",
    "Estado",
    "Estado_EXE",
    "Marcador",
    "Marcador_HT",
    "Corners_total",
    "Tarjetas",
    "Prob_modelo",
    "PROB_Historico",
    "Potencial",
    "ROI_estimado",
    "Ventaja_modelo",
    "Unidades",
    "OTeam",
    "Goles_OTeam",
    "Cuota_Justa",
]


def _clean_series(series: pd.Series | None) -> pd.Series | None:
    if series is None:
        return None
    if series.dtype == object:
        return series.replace("", pd.NA)
    return series


def _coalesce(df: pd.DataFrame, cols: Iterable[str]) -> pd.Series | None:
    out: pd.Series | None = None
    for col in cols:
        if col not in df.columns:
            continue
        series = _clean_series(df[col])
        if series is None:
            continue
        out = series if out is None else out.combine_first(series)
    return out


def reform_maestro(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    d = df.copy()

    d["Match_id"] = _coalesce(d, ["Match_id", "Partido", "ID_partido"])
    d["Local"] = _coalesce(d, ["Local", "Home", "home"])
    d["Visitante"] = _coalesce(d, ["Visitante", "Visita", "Away", "away"])
    d["competition_id"] = _coalesce(d, ["competition_id", "Season_id", "season_id"])
    d["Season_label"] = _coalesce(d, ["Season_label", "season_label"])
    d["home_id"] = _coalesce(d, ["home_id", "homeID", "_home_id"])
    d["away_id"] = _coalesce(d, ["away_id", "awayID", "_away_id"])
    d["Fecha_ejecucion"] = _coalesce(d, ["Fecha_ejecucion", "Fecha"])
    d["Marcador"] = _coalesce(d, ["Marcador", "Marcador_FT"])
    d["Marcador_HT"] = _coalesce(d, ["Marcador_HT", "Marcador_ht"])
    d["Prob_modelo"] = _coalesce(d, ["Prob_modelo", "Probabilidad"])
    d["market_group"] = _coalesce(d, ["market_group", "Market_group"])
    if "market_group" in d.columns:
        d["market_group"] = d["market_group"].fillna(d.get("Mercado"))
    d["Telegram_estado"] = _coalesce(d, ["Telegram_estado", "Telegram_status"])
    d["Telegram_message_id"] = _coalesce(d, ["Telegram_message_id", "Telegram_msg_id"])

    if "Corners_total" in d.columns:
        if "Corners_local" in d.columns or "Corners_visitante" in d.columns:
            base = pd.to_numeric(d.get("Corners_local"), errors="coerce").fillna(0)
            base += pd.to_numeric(d.get("Corners_visitante"), errors="coerce").fillna(0)
            base = base.replace(0, pd.NA)
            d["Corners_total"] = d["Corners_total"].fillna(base)
    else:
        if "Corners_local" in d.columns or "Corners_visitante" in d.columns:
            base = pd.to_numeric(d.get("Corners_local"), errors="coerce").fillna(0)
            base += pd.to_numeric(d.get("Corners_visitante"), errors="coerce").fillna(0)
            d["Corners_total"] = base.replace(0, pd.NA)

    for col in CANONICAL_COLUMNS:
        if col not in d.columns:
            d[col] = pd.NA

    return d[CANONICAL_COLUMNS]


def ensure_maestro_aliases(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    d = df.copy()

    if "Partido" not in d.columns and "Match_id" in d.columns:
        d["Partido"] = d["Match_id"]
    if "ID_partido" not in d.columns and "Match_id" in d.columns:
        d["ID_partido"] = d["Match_id"]
    if "Visita" not in d.columns and "Visitante" in d.columns:
        d["Visita"] = d["Visitante"]
    if "Season_id" not in d.columns and "competition_id" in d.columns:
        d["Season_id"] = d["competition_id"]
    if "season_id" not in d.columns and "competition_id" in d.columns:
        d["season_id"] = d["competition_id"]
    if "season_label" not in d.columns and "Season_label" in d.columns:
        d["season_label"] = d["Season_label"]
    if "homeID" not in d.columns and "home_id" in d.columns:
        d["homeID"] = d["home_id"]
    if "awayID" not in d.columns and "away_id" in d.columns:
        d["awayID"] = d["away_id"]
    if "Probabilidad" not in d.columns and "Prob_modelo" in d.columns:
        d["Probabilidad"] = d["Prob_modelo"]
    if "Marcador_FT" not in d.columns and "Marcador" in d.columns:
        d["Marcador_FT"] = d["Marcador"]

    return d
