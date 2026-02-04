import requests
import pandas as pd
from typing import Optional, Dict, Any
from app.common.config import settings

API_BASE = "https://api.football-data-api.com"
KEY = settings.footystats_key

def _get(path: str, params: Dict[str, Any]) -> dict:
    q = {"key": KEY, **params}
    r = requests.get(f"{API_BASE}{path}", params=q, timeout=20)
    r.raise_for_status()
    
    payload = r.json()
    if not payload.get("success", False):
        raise ValueError(f"API error {path}: {payload}")
    return payload

def _split_year(y: Any):
    s = str(y).strip()
    if len(s) == 8 and s.isdigit():
        return int(s[:4]), int(s[4:]), f"{s[:4]}/{s[4:]}"
    if "/" in s:
        a, b = s.split("/")
        return int(a), int(b), f"{a}/{b}"
    return int(s[:4]), None, str(s[:4])

def fetch_league_index() -> pd.DataFrame:
    """/league-list → DataFrame con season_id y metadatos básicos (sin comp_id)."""
    payload = _get("/league-list", {})
    rows = []
    for lg in payload.get("data", []):
        country = lg.get("country")
        lname   = lg.get("league_name") or lg.get("name")
        full    = lg.get("name")
        for s in lg.get("season", []):
            sid = s.get("id")
            y0, y1, label = _split_year(s.get("year"))
            rows.append({
                "season_id": sid,
                "league_name": lname,
                "league_full_name": full,
                "country": country,
                "season_year_start": y0,
                "season_year_end": y1,
                "season_label": label,
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["season_year_start"] = df["season_year_start"].astype("Int64")
        df["season_year_end"]   = df["season_year_end"].astype("Int64")
    return df

def resolve_competition_id_for_season(season_id: int) -> Optional[int]:
    """Intenta obtener competition_id para un season_id desde varios endpoints."""
    # 1) league-stats
    try:
        p = _get("/league-stats", {"season_id": season_id})
        # muchos payloads vienen con un objeto de liga que incluye competition_id
        # busca en data el primer campo que lo tenga
        for item in p.get("data", []):
            cid = item.get("competition_id") or item.get("league_id") or item.get("id")
            if cid:
                return int(cid)
    except Exception:
        pass
    # 2) league-teams (fallback)
    try:
        p = _get("/league-teams", {"season_id": season_id})
        for item in p.get("data", []):
            cid = item.get("competition_id") or item.get("league_id") or item.get("id_competition")
            if cid:
                return int(cid)
    except Exception:
        pass
    # 3) matches (último recurso)
    try:
        p = _get("/matches", {"season_id": season_id, "page": 1})
        data = p.get("data", [])
        if data:
            # toma competition_id del primer partido que lo tenga
            first = next((m for m in data if m.get("competition_id")), None)
            if first:
                return int(first["competition_id"])
    except Exception:
        pass
    return None

def build_season_to_competition_map(year_filter: int = 2025,
                                    limit: Optional[int] = None) -> pd.DataFrame:
    """
    Devuelve DF con season_id → competition_id + metadatos,
    filtrado solo a temporadas cuyo season_year_start == year_filter.
    """
    idx = fetch_league_index()
    if year_filter:
        idx = idx[idx["season_year_start"] == year_filter]

    if limit:
        idx = idx.head(limit)

    comp_ids = {}
    for sid in idx["season_id"].dropna().astype(int):
        if sid not in comp_ids:
            comp_ids[sid] = resolve_competition_id_for_season(sid)

    idx["Competition_id"] = idx["season_id"].map(comp_ids)
    return idx[[
        "competition_id", "season_id", "league_name", "league_full_name",
        "country", "season_year_start", "season_year_end", "season_label"
    ]]
# 1) Traer índice (season_id, liga, país, años) + competition_id resuelto
df_map = build_season_to_competition_map()  # quita el limit para todo
# print(df_map)

# 2) Enriquecer partidos con competition_id
# df_matches = DataFrame de /matches que tenga season_id
# df_enriched = df_matches.merge(df_map[["season_id","competition_id","league_name","country"]],
#                                on="season_id", how="left")
