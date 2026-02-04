from __future__ import annotations

import argparse
from typing import Any, Dict

import numpy as np
import pandas as pd

from app.ingest.listado_ligas import enrich_liga_cols
from app.ingest.normalizacion.normalizacion_corners import corners_to_df
from app.ingest.historico.corners_contexto import enriquecer_corners


def _expected_corners(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula una expectativa simple de córners por equipo usando promedios propios y
    promedio de córners concedidos por el rival. Incluye fallback a datos del partido
    actual si los promedios históricos no existen (evita dejar todo en 0/NaN).
    """
    d = df.copy()

    def _num(col: str) -> pd.Series:
        if col in d.columns:
            return pd.to_numeric(d[col], errors="coerce")
        return pd.Series(np.nan, index=d.index)

    local_for = _num("local_team_corners_avg")
    visita_for = _num("visita_team_corners_avg")
    local_against = _num("local_team_corners_against_avg")
    visita_against = _num("visita_team_corners_against_avg")
    # Usa primero el histórico; si falta, toma datos del partido actual (si existen).
    team_a = _num("team_a_corners").fillna(_num("home_ft_corners"))
    team_b = _num("team_b_corners").fillna(_num("away_ft_corners"))

    # Fallback: usa datos del partido si los promedios faltan
    local_for = local_for.fillna(team_a)
    visita_for = visita_for.fillna(team_b)

    d["exp_local"] = local_for - visita_against
    d["exp_visita"] = visita_for - local_against
    d["exp_diff"] = d["exp_local"] - d["exp_visita"]
    return d


def build_corners_winner_df(matches: list[dict], diff_min: float = 0.0) -> pd.DataFrame:
    """
    Genera picks para Winner de córners (1X2) usando expectativa de córners por equipo.
    diff_min: umbral mínimo de diferencia esperada para emitir pick.
    """
    base = corners_to_df(matches).copy()
    if base.empty:
        return pd.DataFrame(columns=[
            "Partido", "Season_id", "Hora", "Pais", "Liga", "Local", "Visitante",
            "Mercado", "ODDS", "Exp_diff"
        ])

    df = enriquecer_corners(base)
    # Alinear IDs: si home_id/away_id vienen vacíos pero existe homeID/awayID, rellenar.
    if "home_id" in df.columns and "homeID" in df.columns:
        df["home_id"] = df["home_id"].fillna(df["homeID"])
    if "away_id" in df.columns and "awayID" in df.columns:
        df["away_id"] = df["away_id"].fillna(df["awayID"])
    df = enrich_liga_cols(df, competition_id_col="competition_id", season_id_col="season_id")
    df = df.rename(columns={
        "match_id": "Partido",
        "season_id": "Season_id",
        "home": "Local",
        "away": "Visitante",
        "kickoff_local_cdmx": "Hora",
    })
    # Fallback: si Season_id está vacío, usa competition_id
    if "Season_id" in df.columns and "competition_id" in df.columns:
        df["Season_id"] = df["Season_id"].fillna(df["competition_id"])

    df = _expected_corners(df)

    # Selección 1X2 por diferencia esperada
    picks = []
    for _, row in df.iterrows():
        diff = row.get("exp_diff")
        if pd.isna(diff):
            diff = 0.0  # sin filtro: considera el partido con expectativa neutra
        if diff >= diff_min:
            ganador = "Local corners"
            odds = row.get("odds_corners_1")
        elif diff <= -diff_min:
            ganador = "Visitante corners"
            odds = row.get("odds_corners_2")
        else:
            ganador = "Empate corners"
            odds = row.get("odds_corners_x")

        picks.append({
            "Partido": row.get("Partido"),
            "Season_id": row.get("Season_id"),
            "competition_id": row.get("competition_id"),
            "Hora": row.get("Hora"),
            "Pais": row.get("Pais"),
            "Liga": row.get("Liga"),
            "Local": row.get("Local"),
            "Visitante": row.get("Visitante"),
            "Mercado": ganador,
            "ODDS": odds,
            "Exp_diff": round(diff, 2) if not pd.isna(diff) else None,
            "Corners_local_exp": row.get("exp_local"),
            "Corners_visita_exp": row.get("exp_visita"),
        })

    out = pd.DataFrame(picks)
    if out.empty:
        return out

    out["ODDS"] = pd.to_numeric(out["ODDS"], errors="coerce")
    # Mantén también picks sin odds para observar el comportamiento
    out = out.sort_values("Exp_diff", ascending=False)
    return out.reset_index(drop=True)


def build_arg_parser(default_diff: float) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Winner de córners (1X2)")
    p.add_argument("--diff-min", type=float, default=default_diff,
                   help="Diferencia mínima esperada de córners para emitir pick")
    return p


def main():
    parser = build_arg_parser(0.5)
    args = parser.parse_args()
    # Entrada vacía por CLI interactivo no implementada; este main es placeholder.
    print("Este script espera ser llamado desde orquesta con los partidos del día.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
