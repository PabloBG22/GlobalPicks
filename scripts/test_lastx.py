import os
import sys

import pandas as pd
import requests

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from app.common.config import settings


def fetch_lastx(team_id: int) -> dict:
    url = f"{settings.footystats_url}/lastx"
    params = {
        "key": settings.footystats_key,
        "team_id": team_id,
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def stats_to_dataframe(payload: dict) -> pd.DataFrame:
    if not isinstance(payload, dict):
        raise ValueError("Formato inesperado de lastx: se esperaba dict")

    data = payload.get("data")
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        records = [data]
    else:
        records = [payload]

    rows = []
    for record in records:
        base = {
            "team_id": record.get("id"),
            "team_name": record.get("name") or record.get("full_name"),
            "season": record.get("season"),
            "country": record.get("country"),
            "last_x": record.get("last_x_match_num"),
        }
        stats = record.get("stats") or {}
        flat_stats = {k: stats.get(k) for k in stats.keys()}
        rows.append({**base, **flat_stats})

    return pd.DataFrame(rows)


def main():
    team_id = 59  # Barcelona
    try:
        payload = fetch_lastx(team_id)
        df_stats = stats_to_dataframe(payload)
        salida = os.path.join(PROJECT_ROOT, f"lastx_team_{team_id}.xlsx")
        df_stats.to_excel(salida, index=False)
        print(f"✅ Archivo generado: {salida} (filas: {len(df_stats)})")
    except Exception as exc:
        print(f"❌ Error consultando lastx: {exc}")


if __name__ == "__main__":
    main()
