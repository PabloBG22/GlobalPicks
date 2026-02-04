from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from app.ingest.listado_ligas import enrich_liga_cols
from app.ingest.referees import get_referee_stats


def _prepare_matches(today_match: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(today_match)
    base_cols = [
        "match_id", "competition_id", "season_id", "home", "away",
        "cards_potential", "referee_id", "kickoff_local_cdmx",
    ]
    if df.empty:
        return pd.DataFrame()
    for col in base_cols:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[base_cols + [c for c in ["team_a_cards_num", "team_b_cards_num"] if c in df.columns]].copy()
    df["cards_potential"] = pd.to_numeric(df["cards_potential"], errors="coerce")
    df["team_a_cards_num"] = pd.to_numeric(df.get("team_a_cards_num"), errors="coerce")
    df["team_b_cards_num"] = pd.to_numeric(df.get("team_b_cards_num"), errors="coerce")
    df["season_id"] = pd.to_numeric(df["season_id"], errors="coerce")
    df["competition_id"] = pd.to_numeric(df["competition_id"], errors="coerce")
    return df


def _merge_league_info(df: pd.DataFrame) -> pd.DataFrame:
    return enrich_liga_cols(df, competition_id_col="competition_id", season_id_col="season_id")


def _enrich_with_referees(df: pd.DataFrame) -> pd.DataFrame:
    if "season_id" not in df.columns:
        df = df.assign(season_id=pd.NA)
    parts = []
    for season_id, chunk in df.groupby("season_id"):
        stats = get_referee_stats(int(season_id)) if pd.notna(season_id) else pd.DataFrame()
        if stats.empty:
            chunk["Referee"] = pd.NA
            chunk["Referee_over_pct"] = np.nan
            chunk["Referee_avg_cards"] = np.nan
            parts.append(chunk)
            continue
        merged = chunk.merge(
            stats[[c for c in stats.columns if c not in {"season_id"}]].reset_index(),
            left_on="referee_id", right_on="referee_id", how="left"
        ).fillna({"Referee": "Sin árbitro"})
        merged = merged.rename(columns={
            "name": "Referee",
            "referee_over_cards_pct": "Referee_over_pct",
            "referee_avg_cards": "Referee_avg_cards",
        })
        parts.append(merged)
    if parts:
        return pd.concat(parts, ignore_index=True)
    return df


def _score_cards(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Prob_modelo_cards"] = df["cards_potential"].fillna(0)
    df["Referee_over_pct"] = pd.to_numeric(df.get("Referee_over_pct"), errors="coerce")
    df["Referee_avg_cards"] = pd.to_numeric(df.get("Referee_avg_cards"), errors="coerce")
    df["Tarjetas_totales"] = (
        df["team_a_cards_num"].fillna(0) + df["team_b_cards_num"].fillna(0)
    )

    df["Potencial_pct"] = (df["cards_potential"].fillna(0) / 10 * 100).clip(lower=0, upper=120)
    df["Referee_over_pct"] = df["Referee_over_pct"].fillna(df["Referee_over_pct"].median())
    df["Indice_modelo"] = (
        0.6 * df["Potencial_pct"] + 0.4 * df["Referee_over_pct"]
    ).round(1)

    df["Prediccion_tarjetas"] = (
        df["cards_potential"].fillna(0) * (df["Referee_over_pct"] / 50.0)
    ).round(2)
    df["Acierto"] = np.where(
        df["Tarjetas_totales"].notna() & (df["Tarjetas_totales"] > 0),
        np.where(df["Tarjetas_totales"] >= df["Prediccion_tarjetas"], "Acierto", "Fallo"),
        ""
    )
    df["Estado"] = np.where(df["Acierto"] == "", "Pendiente", "Evaluado")

    def _clasificar(valor: float) -> str:
        if pd.isna(valor):
            return "Desconocido"
        if valor >= 75:
            return "Alta"
        if valor >= 55:
            return "Media"
        return "Baja"

    df["Confianza"] = df["Indice_modelo"].apply(_clasificar)
    return df


def main_cards_referee(today_match: List[Dict[str, Any]]) -> pd.DataFrame:
    base = _prepare_matches(today_match)
    if base.empty:
        return pd.DataFrame(columns=[
            "Partido", "Liga", "Pais", "Hora",
            "Local", "Visitante", "RefereeID", "Referee",
            "cards_potential", "Referee_over_pct", "Referee_avg_cards",
            "Indice_modelo", "Prediccion_tarjetas", "Confianza"
        ])

    base = _merge_league_info(base)
    enriched = _enrich_with_referees(base)
    scored = _score_cards(enriched)

    scored = scored.rename(columns={
        "cards_potential": "Potencial_cards",
        "kickoff_local_cdmx": "Hora",
        "home": "Local",
        "away": "Visitante",
        "referee_id": "RefereeID",
    })
    cols = [
        "match_id", "Liga", "Pais", "Hora",
        "Local", "Visitante", "RefereeID", "Referee",
        "Potencial_cards", "Referee_over_pct", "Referee_avg_cards",
        "Indice_modelo", "Prediccion_tarjetas", "Tarjetas_totales",
        "Confianza", "Estado", "Acierto"
    ]
    available_cols = [c for c in cols if c in scored.columns]
    scored = scored[available_cols].sort_values("Indice_modelo", ascending=False, na_position="last")
    scored = scored.rename(columns={"match_id": "Partido"})
    if "referee_id" in scored.columns:
        scored = scored.rename(columns={"referee_id": "RefereeID"})
        extra_cols = ["RefereeID"] + [c for c in scored.columns if c != "RefereeID"]
        scored = scored[extra_cols]
    return scored.reset_index(drop=True)
