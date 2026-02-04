import functools
from typing import Dict, Any

import pandas as pd
import requests

from app.common.config import settings


def _request_referees(season_id: int) -> pd.DataFrame:
    url = f"{settings.footystats_url}/league-referees"
    params = {
        "key": settings.footystats_key,
        "season_id": season_id,
        "max_per_page": 500,
    }
    rows = []
    page = 1
    while True:
        params["page"] = page
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        payload = resp.json()
        data = payload.get("data", []) or payload
        if not data:
            break
        rows.extend(data)
        pager = payload.get("pager", {})
        if not pager or page >= pager.get("max_page", page):
            break
        page += 1
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    id_col = next((c for c in ("refereeID", "referee_id", "id") if c in df.columns), None)
    if id_col:
        df = df.rename(columns={id_col: "referee_id"})
    else:
        df["referee_id"] = pd.NA
    numeric_cols = [c for c in df.columns if c not in {"referee_id", "name", "country", "nationality"}]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    percent_cols = [c for c in df.columns if isinstance(c, str) and "percentage" in c]
    if percent_cols:
        df["referee_over_cards_pct"] = df[percent_cols].mean(axis=1)
    avg_cols = [c for c in df.columns if "avg_cards" in str(c).lower()]
    if avg_cols:
        df["referee_avg_cards"] = df[avg_cols].mean(axis=1)
    return df


@functools.lru_cache(maxsize=32)
def get_referee_stats(season_id: int) -> pd.DataFrame:
    df = _request_referees(int(season_id))
    if df.empty:
        return df
    df = df.drop_duplicates(subset=["referee_id"]).set_index("referee_id")
    return df
