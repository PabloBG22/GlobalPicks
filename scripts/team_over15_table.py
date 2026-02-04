import argparse
import math
from datetime import date
from typing import Any, Dict, Optional

import pandas as pd
from tabulate import tabulate

from app.ingest.listado_ligas import enrich_liga_cols
from app.ingest.today_match import get_todays_matches_normalized


def _poisson_p_ge2(lam: Optional[float]) -> Optional[float]:
    """P(X >= 2) for X ~ Poisson(lam)."""
    if lam is None:
        return None
    try:
        l = float(lam)
    except (TypeError, ValueError):
        return None
    if l <= 0:
        return None
    return 1.0 - math.exp(-l) * (1.0 + l)


def build_team_over15_table(fecha: str) -> pd.DataFrame:
    """Devuelve una tabla filtrada con el equipo más probable de marcar 2+ goles por partido."""
    matches = get_todays_matches_normalized(fecha)
    if not matches:
        return pd.DataFrame()
    rows = []

    for m in matches:
        p_home = _poisson_p_ge2(m.get("team_a_xg_prematch"))
        p_away = _poisson_p_ge2(m.get("team_b_xg_prematch"))
        opciones = {"Local": p_home, "Visitante": p_away}
        opciones = {k: v for k, v in opciones.items() if v is not None}
        if not opciones:
            continue

        oteam, prob = max(opciones.items(), key=lambda kv: kv[1])
        prob_pct = prob * 100
        if prob_pct < 70:
            continue

        rows.append(
            {
                "competition_id": m.get("competition_id"),
                "season_id": m.get("season_id"),
                "Hora": m.get("kickoff_local_cdmx"),
                "Pais": "",
                "Liga": "",
                "Local": m.get("home"),
                "Visitante": m.get("away"),
                "Mercado": "Team Over 1.5",
                "OTeam": oteam,
                "Probabilidad": round(prob_pct, 1),
                "Cuota_Justa": round(1.0 / prob, 2) if prob and prob > 0 else None,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["Hora", "Pais", "Liga", "Local", "Visitante", "Mercado", "OTeam", "Probabilidad", "Cuota_Justa"])

    df = pd.DataFrame(rows)
    df = enrich_liga_cols(df, competition_id_col="competition_id", season_id_col="season_id")
    df = df.drop(columns=["competition_id", "season_id"], errors="ignore")
    df = df.sort_values("Probabilidad", ascending=False).reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(description="Tabla de equipos con mayor probabilidad de Over 1.5 goles (anotan 2+).")
    parser.add_argument("--date", dest="fecha", default=date.today().isoformat(), help="Fecha en formato YYYY-MM-DD (por defecto hoy).")
    args = parser.parse_args()

    df = build_team_over15_table(args.fecha)
    if df.empty:
        print(f"No hay partidos con probabilidad > 70% para {args.fecha}")
        return

    print(f"\nTabla Over 1.5 por equipo (>=70%) - {args.fecha}")
    print(tabulate(df, headers="keys", tablefmt="rounded_outline", showindex=False))


if __name__ == "__main__":
    main()
